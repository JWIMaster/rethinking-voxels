"""Focused OpenGL regression for actual voxel LOD/reading and shared-light functions."""
from pathlib import Path
import re
import numpy as np
import moderngl

ROOT = Path(__file__).resolve().parents[1] / 'shaders'

def expand(path):
    text = path.read_text(encoding='utf-8')
    return re.sub(r'^\s*#include\s+"([^"]+)"', lambda m: expand(ROOT / m[1].lstrip('/')), text, flags=re.M)

def function(text, name):
    start = re.search(r'\b(?:bool|float)\s+' + name + r'\s*\(', text).start()
    opening = text.index('{', start)
    depth = 1
    i = opening + 1
    while depth:
        depth += (text[i] == '{') - (text[i] == '}')
        i += 1
    return text[start:i]

NORM = 'float infnorm(vec3 p) { return max(abs(p.x), max(abs(p.y), abs(p.z))); }\n'

def main():
    ctx = moderngl.create_standalone_context(require=430)
    print(ctx.info['GL_RENDERER'], ctx.info['GL_VERSION'])
    rng = np.random.default_rng(81)
    lod_cases = 0
    for size in [(96,64,96), (128,96,128), (256,128,256), (512,128,512)]:
        points = rng.uniform(-1.5,1.5,(300,3)) * np.array(size)
        points = np.concatenate((np.zeros((1,3)), points, [[1e-9,0,0], [-size[0]/2-0.25,0,0], [-size[0]/2,0,0], [size[0]/2,0,0]]))
        inp = np.zeros((len(points),4),dtype='f4'); inp[:,:3] = points
        for detail in range(1,9):
            source = '#version 430\n#define VOXEL_DETAIL_AMOUNT %d\nconst ivec3 voxelVolumeSize = ivec3(%d,%d,%d);\n' % (detail,*size)
            source += NORM + expand(ROOT/'lib/vx/voxelLod.glsl')
            source += '''
layout(local_size_x=1) in;
layout(std430,binding=0) readonly buffer In { vec4 positions[]; };
layout(std430,binding=1) buffer Out { ivec4 values[]; };
void main(){ uint i=gl_GlobalInvocationID.x; vec3 p=positions[i].xyz;
values[i]=ivec4(getVoxelResolution(p),getVoxelWriteLevelCount(abs(p)),int(voxelPositionInBounds(p)),int(voxelCoordInBounds(voxelCellCoord(p)))); }
'''
            shader = ctx.compute_shader(source)
            ib=ctx.buffer(inp.tobytes()); ob=ctx.buffer(reserve=len(inp)*16)
            ib.bind_to_storage_buffer(0); ob.bind_to_storage_buffer(1)
            shader.run(group_x=len(inp)); ctx.memory_barrier()
            actual=np.frombuffer(ob.read(),dtype='i4').reshape(-1,4)
            p=inp[:,:3].astype('f8'); sz=np.array(size)
            read=np.clip(np.floor(-np.log2(np.maximum(np.max(np.abs(p)/(sz-2.01),axis=1),1e-8)))-1,0,detail-1).astype(int)
            write=np.clip(np.floor(-np.log2(np.maximum(np.max(np.abs(p)/sz,axis=1),1e-8))),0,detail).astype(int)
            inside=((p>=-sz/2)&(p<sz/2)).all(axis=1).astype(int)
            np.testing.assert_array_equal(actual[:,0],read)
            np.testing.assert_array_equal(actual[:,1],write)
            np.testing.assert_array_equal(actual[:,2],inside)
            np.testing.assert_array_equal(actual[:,3],inside)
            lod_cases += len(inp)
            shader.release(); ib.release(); ob.release()
    print('PASS finite LOD / negative-floor / bounds:',lod_cases,'queries in 32 real-size/detail configurations')

    size=(32,24,32); detail=8
    head='#version 430\n#define VOXEL_DETAIL_AMOUNT 8\n#define RT_STEPS 64\n#define TRANSLUCENT_LIGHT_TINT\nconst ivec3 voxelVolumeSize=ivec3(32,24,32);\nuniform int frameCounter;\n'+NORM
    reading=expand(ROOT/'lib/vx/voxelReading.glsl')
    atlas=np.empty((size[2],4*size[1],size[0],4),dtype='f4')
    for band in range(4):
        atlas[:,band*size[1]:(band+1)*size[1],:,:]=[100*band+c+1 for c in range(4)]
    tex=ctx.texture3d((size[0],4*size[1],size[2]),4,atlas.tobytes(),dtype='f4')
    tex.filter=(moderngl.LINEAR,moderngl.LINEAR); tex.repeat_x=tex.repeat_y=tex.repeat_z=True; tex.use(0)
    occ=ctx.texture3d(size,1,np.zeros((size[2],size[1],size[0]),dtype='i4').tobytes(),dtype='i4'); occ.bind_to_image(0,read=True,write=False)
    cols=ctx.texture3d((size[0],size[1]*2,size[2]),1,np.zeros((size[2],size[1]*2,size[0]),dtype='i4').tobytes(),dtype='i4'); cols.bind_to_image(1,read=True,write=False)
    source=head+reading+'''
layout(local_size_x=1) in;
layout(std430,binding=0) readonly buffer In { vec4 points[]; };
layout(std430,binding=1) buffer Out { vec4 results[]; };
void main(){ uint i=gl_GlobalInvocationID.x; int level=int(points[i].w); results[i]=vec4(getDistanceFieldAtLevel(points[i].xyz,level),getDistanceField(points[i].xyz),float(getVoxelResolution(points[i].xyz)),getColor(points[i].xyz).a); }
'''
    shader=ctx.compute_shader(source); shader['distanceField'].value=0; shader['voxelCols'].value=1
    if 'occupancyVolume' in shader: shader['occupancyVolume'].value=0
    points=[]
    for level in range(8):
        half=np.array(size)*.5/(2**level)
        for p in [[0,0,0],[0,-half[1]+1e-6,0],[0,half[1]-1e-6,0],[half[0]+.001,0,0],[0,half[1],0]]:
            points.append([*p,level])
    points += [[np.nan,0,0,0], [0,np.inf,0,0]]
    points=np.asarray(points,dtype='f4')
    ib=ctx.buffer(points.tobytes()); ob=ctx.buffer(reserve=len(points)*16)
    ib.bind_to_storage_buffer(0); ob.bind_to_storage_buffer(1)
    for frame in [0,1,2,3]:
        shader['frameCounter'].value=frame; shader.run(group_x=len(points)); ctx.memory_barrier()
        out=np.frombuffer(ob.read(),dtype='f4').reshape(-1,4)
        assert np.isfinite(out).all()
        for i,p in enumerate(points):
            level=int(p[3]); half=np.array(size)*.5/(2**level)
            valid=((p[:3]>=-half)&(p[:3]<half)).all()
            expected=100*(((frame+1)%2)*2+level//4)+level%4+1 if valid else 1
            np.testing.assert_allclose(out[i,0],expected,atol=.002)
    print('PASS atlas:',len(points)*4,'edge/slab/parity/fine-support/nonfinite samples, linear+repeat enabled')
    shader.release(); ib.release(); ob.release()

    source=head+reading+'''
layout(local_size_x=1) in;
layout(std430,binding=0) readonly buffer In { vec4 args[]; };
layout(std430,binding=1) buffer Out { vec4 results[]; };
void main(){ uint i=gl_GlobalInvocationID.x; vec3 p=args[2*i].xyz,d=args[2*i+1].xyz,n; results[3*i]=vec4(rayTrace(p,d,0.5),1); results[3*i+1]=coneTrace(p,d,0.1,0.5); results[3*i+2]=voxelTrace(p,d,n,1); }
'''
    shader=ctx.compute_shader(source); shader['distanceField'].value=0; shader['occupancyVolume'].value=0
    tests=[([0,0,0],[0,0,0]),([0,0,0],[10,0,0]),([0,0,0],[0,-10,0]),([0,0,0],[0,0,10]),([0,0,0],[20,0,0]),([-16.1,0,0],[10,0,0]),([0,0,0],[10,10,10])]
    tests += [([0,0,0],[np.nan,1,0]), ([0,0,0],[0,np.inf,1])]
    args=np.zeros((len(tests)*2,4),dtype='f4')
    for i,(p,d) in enumerate(tests): args[2*i,:3]=p; args[2*i+1,:3]=d
    ib=ctx.buffer(args.tobytes()); ob=ctx.buffer(reserve=len(tests)*3*16)
    ib.bind_to_storage_buffer(0); ob.bind_to_storage_buffer(1)
    shader['frameCounter'].value=0; shader.run(group_x=len(tests)); ctx.memory_barrier()
    out=np.frombuffer(ob.read(),dtype='f4').reshape(len(tests),3,4)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[0,0,:3],0)
    np.testing.assert_array_equal(out[0,1,:],1)
    np.testing.assert_array_equal(out[0,2,:],-10000)
    np.testing.assert_allclose(out[5,0,:3],[-6.1,0,0],atol=1e-6)
    np.testing.assert_array_equal(out[5,2,:],-10000)
    for i in [7,8]:
        np.testing.assert_array_equal(out[i,0,:3],0)
        np.testing.assert_array_equal(out[i,1,:],1)
        np.testing.assert_array_equal(out[i,2,:],-10000)
    print('PASS',len(tests)*3,'actual ray/cone/DDA calls: zero, nonfinite, axis-parallel, bounds escape and miss sentinel')
    # Occupied signed distance must stop rays, never advance them backwards.
    negative=ctx.texture3d((size[0],4*size[1],size[2]),4,np.full(atlas.shape,-.125,dtype='f4').tobytes(),dtype='f4')
    negative.filter=(moderngl.LINEAR,moderngl.LINEAR); negative.use(0)
    occupancy=np.zeros((size[2],size[1],size[0]),dtype='i4')
    occupancy[16,12,18]=1
    occ.write(occupancy.tobytes())
    shader.run(group_x=len(tests)); ctx.memory_barrier()
    blocked=np.frombuffer(ob.read(),dtype='f4').reshape(len(tests),3,4)
    assert np.isfinite(blocked).all()
    assert blocked[1,0,0]>=0 and blocked[1,0,0]<.01
    assert blocked[1,1,3]==0
    np.testing.assert_allclose(blocked[1,2,:],[2,0,0,1],atol=1e-5)
    print('PASS signed occupied distance: no backwards ray movement; cone visibility0; exact axis-parallel DDA hit atx2')
    shader.release(); ib.release(); ob.release(); tex.release(); negative.release(); occ.release(); cols.release()

    prepare=(ROOT/'program/prepare4_csh.glsl').read_text()
    hashcode=expand(ROOT/'lib/vx/positionHashing.glsl')
    claim=function(prepare,'claimLightCell')
    source='#version 430\nconst ivec3 voxelVolumeSize=ivec3(32,24,32);\n#define LIGHT_DEDUP_CAPACITY 1024\nshared uint lightHashMap[1024];\n'+hashcode+claim+'''
layout(local_size_x=1024) in;
layout(std430,binding=0) readonly buffer In { ivec4 coords[]; };
layout(std430,binding=1) buffer Out { int claimed[]; };
void main(){ uint i=gl_LocalInvocationIndex; lightHashMap[i]=0u; barrier(); claimed[i]=claimLightCell(coords[i].xyz)?1:0; }
'''
    shader=ctx.compute_shader(source)
    flat=rng.choice(np.prod(size),512,replace=False)
    coords=np.zeros((1024,4),dtype='i4')
    for i,k in enumerate(flat):
        p=[k%size[0],k//size[0]%size[1],k//(size[0]*size[1])]
        coords[2*i,:3]=p; coords[2*i+1,:3]=p
    ib=ctx.buffer(coords.tobytes()); ob=ctx.buffer(reserve=1024*4)
    ib.bind_to_storage_buffer(0); ob.bind_to_storage_buffer(1)
    for trial in range(5):
        shader.run(); ctx.memory_barrier()
        claimed=np.frombuffer(ob.read(),dtype='i4')
        np.testing.assert_array_equal(claimed.reshape(-1,2).sum(axis=1),1)
        assert claimed.sum()==512
    print('PASS exact shared dedup: all512 distinct cells retained,1024 concurrent duplicate attempts,5runs')
    shader.release(); ib.release(); ob.release()

    falloff=function(prepare,'distanceFalloff')
    for r2 in [0,1]:
        source='#version 430\n#define LIGHT_TRACE_LENGTH 32.0\n#define R2_FALLOFF %d\n'%r2+falloff+'''
layout(local_size_x=1) in; layout(std430,binding=0) readonly buffer In { float ratios[]; }; layout(std430,binding=1) buffer Out { float values[]; }; void main(){uint i=gl_GlobalInvocationID.x;values[i]=distanceFalloff(ratios[i]);}
'''
        shader=ctx.compute_shader(source)
        vals=np.asarray([-1,0,.1,.99,1,1.1,np.nan,np.inf],dtype='f4')
        ib=ctx.buffer(vals.tobytes()); ob=ctx.buffer(reserve=vals.nbytes)
        ib.bind_to_storage_buffer(0); ob.bind_to_storage_buffer(1)
        shader.run(group_x=len(vals)); ctx.memory_barrier()
        out=np.frombuffer(ob.read(),dtype='f4')
        assert np.isfinite(out).all() and (out>=0).all()
        np.testing.assert_array_equal(out[4:],0)
        shader.release(); ib.release(); ob.release()
    assert 'bool validData = (normalDepthData.a' in prepare
    assert 'bool foundValidData' in prepare and 'validData = true;' in prepare
    assert '-log2(infnorm(abs(vxPos) / voxelVolumeSize) - 0.5)' not in prepare
    assert 'imageLoad(colorimg11' not in prepare
    assert 'rayHit0.w > 0.0' in prepare
    assert 'globalLightHashMap' not in prepare
    print('PASS 16 finite falloff cases and source guards: rescued geometry, immutable candidates, no missing hit registration')
    print('ALL VOXEL LOD REGRESSIONS PASS')

if __name__=='__main__': main()
