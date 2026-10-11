"""Numerical RTX/OpenGL regressions for immutable rolling and screen histories.
Run: python tools/test_voxel_history.py. No game/profile writes.
"""
from pathlib import Path
import re
import numpy as np
import moderngl
from test_kino_streak import source
ROOT = Path(__file__).resolve().parents[1] / 'shaders'
V = (16, 8, 16)
ctx = moderngl.create_standalone_context(require=430)

def compute(name, options=None, bindings=None, volume=V):
    text = source(ROOT / name, options)
    text = re.sub(r'const ivec3 voxelVolumeSize = ivec3\([^;]+;', 'const ivec3 voxelVolumeSize = ivec3(' + ', '.join(map(str,volume)) + ');', text)
    # ModernGL misclassifies INT_IMAGE_3D uniform setters on this driver. Iris
    # uses glUniform1i; explicit fixture-only bindings avoid that harness defect.
    for name, unit in (bindings or {}).items():
        text = re.sub(r'layout\(([^)]*)\)([^;\n]*\b' + name + r'\s*;)', lambda m: 'layout(' + m.group(1) + ', binding=' + str(unit) + ')' + m.group(2), text)
    return ctx.compute_shader(text)

def uniform(p, name, value):
    if name in p: p[name].value = value

def state(p, frame=3, delta=(0, 0, 0), view=None):
    uniform(p, 'frameCounter', frame)
    uniform(p, 'cameraPositionInt', delta)
    uniform(p, 'previousCameraPositionInt', (0, 0, 0))
    if view:
        uniform(p, 'viewWidth', float(view[0])); uniform(p, 'viewHeight', float(view[1]))

def image(p, name, tex, unit):
    if name in p:
        tex.bind_to_image(unit)

def sampler(p, name, tex, unit):
    if name in p:
        tex.use(unit); p[name].value = unit

def matrix(p, name, value):
    if name in p: p[name].write(np.asarray(value, dtype='f4').T.tobytes())

def barrier(): ctx.memory_barrier(moderngl.ALL_BARRIER_BITS)

def rolling():
    w,h,d=V
    colors=np.arange(d*2*h*w*4,dtype='f4').reshape(d,2*h,w,4) % 97
    colors=colors.astype('f2')
    zz,yy,xx=np.indices((d,h,w))
    lights=np.stack([xx-w//2,yy-h//2,zz-d//2,1+xx+17*yy+139*zz],axis=-1).astype('i2')
    live=ctx.texture3d((w,2*h,d),4,dtype='f2'); snap=ctx.texture3d((w,2*h,d),4,dtype='f2')
    li=ctx.texture3d((w,h,d),4,dtype='i2'); ls=ctx.texture3d((w,h,d),4,dtype='i2')
    bindings={'irradianceCacheI':0,'irradianceSnapshotI':1,'lightStorage':2,'lightStorageSnapshotI':3}
    copy=compute('world0/shadowcomp1.csh',bindings=bindings); shift=compute('world0/shadowcomp1_a.csh',bindings=bindings)
    cases=[(3,(0,0,0)),(3,(1,0,0)),(3,(-1,0,0)),(3,(0,1,0)),(3,(0,-1,0)),(3,(0,0,1)),(3,(0,0,-1)),(3,(1,-1,1)),(3,(16,0,0)),(0,(0,0,0))]
    for frame,delta in cases:
        live.write(colors.tobytes()); li.write(lights.tobytes())
        for p in (copy,shift):
            state(p,frame,delta)
            for n,t,u in [('irradianceCacheI',live,0),('irradianceSnapshotI',snap,1),('lightStorage',li,2),('lightStorageSnapshotI',ls,3)]: image(p,n,t,u)
        copy.run(2,1,2);barrier(); snapshot_before=snap.read();light_before=ls.read()
        shift.run(2,1,2);barrier()
        expected=np.zeros_like(colors); el=np.zeros_like(lights)
        valid=frame>0 and all(abs(delta[i])<V[i] for i in range(3))
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    px,py,pz=x+delta[0],y+delta[1],z+delta[2]
                    if valid and 0<=px<w and 0<=py<h and 0<=pz<d:
                        expected[z,y,x]=(colors[pz,py,px].astype('f4')*np.float32(.99)).astype('f2')
                        expected[z,y+h,x]=colors[pz,py+h,px]
                        el[z,y,x]=lights[pz,py,px]
                        el[z,y,x,:3]-=delta
        actual=np.frombuffer(live.read(),dtype='f2').reshape(colors.shape)
        # NVIDIA imageStore RGBA16F conversion can truncate instead of NumPy's
        # nearest rounding. Decay permits one half ULP; untouched bank is exact.
        np.testing.assert_allclose(actual[:,:h],expected[:,:h],rtol=.0011,atol=1e-5)
        np.testing.assert_array_equal(actual[:,h:],expected[:,h:])
        np.testing.assert_array_equal(np.frombuffer(li.read(),dtype='i2').reshape(lights.shape),el)
        assert snap.read()==snapshot_before and ls.read()==light_before
    print(f'PASS rolling: {len(cases)} integer shifts/reset/teleport cases, both banks and world-aligned light lists; snapshots unchanged')

def gi_diffusion():
    # Extract the actual air-cell diffusion loop and its bounded history helper.
    s=(ROOT/'program/shadowcomp1.glsl').read_text()
    helper=s[s.index('    vec4 ReadGIHistory('):s.index('    #include "/lib/vx/SSBOs.glsl"',s.index('    vec4 ReadGIHistory('))]
    start=s.index('                vec4 GILight = vec4(0.0);')
    end=s.index('                imageStore(irradianceCacheI, coords, GILight);',start)+len('                imageStore(irradianceCacheI, coords, GILight);')
    body=s[start:end]
    shader='''#version 430
const ivec3 voxelVolumeSize=ivec3(16,8,16);
layout(local_size_x=8,local_size_y=8,local_size_z=8) in;
layout(rgba16f, binding=1) uniform writeonly image3D irradianceCacheI;
layout(rgba16f, binding=0) uniform readonly image3D irradianceSnapshotI;
bool VoxelHistoryCoordValid(ivec3 p) { return all(greaterThanEqual(p,ivec3(0))) && all(lessThan(p,voxelVolumeSize)); }
'''+helper+'\nvoid main(){ivec3 coords=ivec3(gl_GlobalInvocationID);'+body+'}\n'
    p=ctx.compute_shader(shader)
    w,h,d=V
    data=np.random.default_rng(6).uniform(0,8,(d,2*h,w,4)).astype('f2')
    data[:,h:]=60000 # distinct direct-light bank must never bleed into GI
    snap=ctx.texture3d((w,2*h,d),4,data.tobytes(),dtype='f2')
    out=ctx.texture3d((w,2*h,d),4,dtype='f2')
    image(p,'irradianceSnapshotI',snap,0);image(p,'irradianceCacheI',out,1)
    expected=np.zeros((d,h,w,4),dtype='f4')
    for axis in (0,1,2):
        left=[slice(None)]*4;right=left.copy();left[axis]=slice(1,None);right[axis]=slice(None,-1)
        expected[tuple(left)]+=np.float32(.16667)*data[:,:h][tuple(right)].astype('f4')
        expected[tuple(right)]+=np.float32(.16667)*data[:,:h][tuple(left)].astype('f4')
    before=snap.read(); results=[]
    for _ in range(4):
        p.run(2,1,2);barrier(); actual=np.frombuffer(out.read(),dtype='f2').reshape(d,2*h,w,4)[:,:h]
        np.testing.assert_allclose(actual,expected.astype('f2'),rtol=.002,atol=.008)
        results.append(actual.copy());assert snap.read()==before
    for a in results[1:]:np.testing.assert_array_equal(a,results[0])
    print('PASS actual GI diffusion: CPU Jacobi match, four bit-identical repeats, packed-bank edge isolation')

def screen_snapshots():
    w,h=19,13
    values=np.arange(w*h*4,dtype='f4').reshape(h,w,4)%29
    candidates=np.arange(w*h*4,dtype='i2').reshape(h,w,4)
    for highlight in (False,True):
        p=compute('world0/prepare4.csh',{'BLOCKLIGHT_HIGHLIGHT':highlight}, {'colorimg12':0,'colorimg14':1,'colorimg11':2,'diffuseHistoryI':3,'specularHistoryI':4,'candidateHistoryI':5})
        diffuse=ctx.texture((w,h),4,values.astype('f2').tobytes(),dtype='f2')
        spec=ctx.texture((w,h),4,(values*.5).astype('f2').tobytes(),dtype='f2')
        candidate=ctx.texture((w,h),4,candidates.tobytes(),dtype='i2')
        dh=ctx.texture((w,h),4,dtype='f2');sh=ctx.texture((w,h),4,dtype='f2');ch=ctx.texture((w,h),4,dtype='i2')
        for n,t,u in [('colorimg12',diffuse,0),('colorimg14',spec,1),('colorimg11',candidate,2),('diffuseHistoryI',dh,3),('specularHistoryI',sh,4),('candidateHistoryI',ch,5)]:image(p,n,t,u)
        for frame in (3,0):
            state(p,frame,view=(w,h));p.run(2,1,1);barrier()
            np.testing.assert_array_equal(np.frombuffer(dh.read(),dtype='f2').reshape(h,w,4),values if frame else 0)
            np.testing.assert_array_equal(np.frombuffer(ch.read(),dtype='i2').reshape(h,w,4),candidates if frame else 0)
            if highlight:np.testing.assert_array_equal(np.frombuffer(sh.read(),dtype='f2').reshape(h,w,4),values*.5 if frame else 0)
    print('PASS screen snapshots: odd viewport, candidate/diffuse/specular copying and invalid-history clears')

def temporal_gating():
    w,h=19,13
    p=compute('world0/prepare4_b.csh',{'BLOCKLIGHT_HIGHLIGHT':False,'BLOCKLIGHT_RESOLUTION':3}, {'colorimg12':0,'diffuseHistoryI':1})
    near,far=.1,100.
    P=np.array([[1,0,0,0],[0,1,0,0],[0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]],dtype='f4')
    def depth(z):return float((P[2,2]*(-z)+P[2,3])/z*.5+.5)
    D=depth(3)
    normal=np.zeros((h,w,4),dtype='f4');normal[:,:,2]=1;normal[:,:,3]=1-D
    oldnormal=np.zeros((h,w,4),dtype='f4');oldnormal[:,:,1:3]=.5;oldnormal[:,:,3]=1
    rng=np.indices((h,w)).sum(axis=0)%2
    raw=np.full((h,w,4),1,dtype='f2');raw[:,:,:3]=(rng[:,:,None]*9+.5)
    textures={
        'colortex10':ctx.texture((w,h),4,raw.tobytes(),dtype='f2'),
        'colortex8':ctx.texture((w,h),4,normal.tobytes(),dtype='f4'),
        'colortex4':ctx.texture((w,h),4,oldnormal.tobytes(),dtype='f4'),
        'colortex1':ctx.texture((w,h),1,dtype='f4'),
        'irradianceCache':ctx.texture3d((16,16,16),4,bytes(16*16*16*4*2),dtype='f2'),
    }
    out=ctx.texture((w,h),4,dtype='f2');history=ctx.texture((w,h),4,dtype='f2')
    image(p,'colorimg12',out,0);image(p,'diffuseHistoryI',history,1)
    for unit,(name,tex) in enumerate(textures.items()):
        tex.filter=(moderngl.NEAREST,moderngl.NEAREST);sampler(p,name,tex,unit)
    matrix(p,'gbufferModelViewInverse',np.eye(4));matrix(p,'gbufferProjectionInverse',np.linalg.inv(P));matrix(p,'gbufferPreviousProjection',P);matrix(p,'gbufferPreviousModelView',np.eye(4))
    state(p,view=(w,h))
    def run(old_depth,prior,frame=3):
        state(p,frame,view=(w,h))
        textures['colortex1'].write(np.full((h,w),old_depth,dtype='f4').tobytes())
        history.write(np.full((h,w,4),prior,dtype='f2').tobytes())
        before=history.read();p.run(2,1,1);barrier();assert history.read()==before
        a=np.frombuffer(out.read(),dtype='f2').reshape(h,w,4).copy();assert np.isfinite(a).all();return a
    valid_a,valid_b=run(D,5),run(D,8)
    assert np.max(abs(valid_a[:,:,:3].astype('f4')-valid_b[:,:,:3]))>.05, 'valid old lighting should contribute'
    invalid_a,invalid_b=run(depth(30),5),run(depth(30),8)
    np.testing.assert_array_equal(invalid_a,invalid_b)
    np.testing.assert_array_equal(run(D,5,0),run(D,8,0))
    print('PASS actual temporal reconstruct: same-normal depth disocclusion rejects old light; valid history contributes; reset ignores old light; odd viewport finite; snapshots unchanged')

def bank_sampling():
    helper=(ROOT/'lib/vx/irradianceCache.glsl').read_text()
    data=np.ones((4,8,4,4),dtype='f4');data[:,4:,:,:3]=100
    tex=ctx.texture3d((4,8,4),4,data.tobytes(),dtype='f4');tex.filter=(moderngl.LINEAR,moderngl.LINEAR)
    for snapshot in (False,True):
        text='#version 430\nconst ivec3 voxelVolumeSize=ivec3(4);\n'+('#define IRRADIANCE_SNAPSHOT_READER\n' if snapshot else '')+helper+'''\nlayout(local_size_x=1) in;
layout(std430,binding=0) buffer Result { vec4 result[2]; };
void main(){result[0]=vec4(readIrradianceCache(vec3(0,1.99,0),vec3(0,1,0)),1);result[1]=vec4(readSurfaceVoxelBlocklight(vec3(0,-1.99,0),vec3(0,-1,0)),1);}
'''
        p=ctx.compute_shader(text);buf=ctx.buffer(reserve=32);buf.bind_to_storage_buffer(0)
        sampler(p,'irradianceSnapshot' if snapshot else 'irradianceCache',tex,0);p.run();barrier()
        values=np.frombuffer(buf.read(),dtype='f4').reshape(2,4)
        np.testing.assert_allclose(values[0,:3],1,rtol=1e-6)
        np.testing.assert_allclose(values[1,:3],100*np.log(100*np.sqrt(3)+1)/(100*np.sqrt(3)),rtol=1e-5)
    print('PASS GI/direct sampler bank isolation after positive/negative normal bias; live and snapshot sampler aliases')

def palette_extension():
    w,h,d=V
    p=compute('world0/shadowcomp.csh', bindings={'occupancyVolume':0,'distanceFieldI':1,'voxelCols':2,'voxelLightDataI':3})
    occupancy=ctx.texture3d((w,2*h,d),1,bytes(w*2*h*d*4),dtype='i4')
    df=ctx.texture3d((w,4*h,d),4,bytes(w*4*h*d*4*2),dtype='f2')
    palette=ctx.texture3d((w,4*h,d),1,dtype='i4')
    light=ctx.texture3d((w,4*h,d),1,bytes(w*4*h*d*4),dtype='u4')
    for name,tex,unit in [('occupancyVolume',occupancy,0),('distanceFieldI',df,1),('voxelCols',palette,2),('voxelLightDataI',light,3)]:image(p,name,tex,unit)
    state(p)
    data=np.zeros((d,4*h,w),dtype='i4');z,y,x=8,4,7
    data[z,2*y,x]=(1<<28)+20+(40<<13);data[z,2*y+1,x]=60+(1<<23)+(1000<<13)
    results=[]
    for _ in range(5):
        palette.write(data.tobytes());p.run(2,1,2);barrier()
        a=np.frombuffer(palette.read(),dtype='i4').reshape(d,4*h,w).copy()
        assert a[z,2*y,x]==data[z,2*y,x]
        for q in (x-1,x+1):
            assert a[z,2*y+1,q]>>23==1 and a[z,2*y,q]&(1<<28)==0
        for q in (x-2,x+2):assert a[z,2*y+1,q]==0
        results.append(a)
    for a in results[1:]:np.testing.assert_array_equal(a,results[0])
    print('PASS actual palette extension: source provenance preserved; one-cell donors only; no two-cell flood across workgroups; five exact repeats')

def entity_clumping():
    w,h,d=40,8,16
    p=compute('world0/shadowcomp_a.csh', bindings={'occupancyVolume':0,'voxelLightDataI':1},volume=(w,h,d))
    occupancy=ctx.texture3d((w,2*h,d),1,dtype='i4');light=ctx.texture3d((w,4*h,d),1,dtype='u4')
    image(p,'occupancyVolume',occupancy,0);image(p,'voxelLightDataI',light,1)
    flags=1|(1<<8)|(2<<22)|(1<<27)|(3<<28)
    def run(nodes):
        bits=np.zeros((d,2*h,w),dtype='i4');records=np.zeros((d,4*h,w),dtype='u4')
        for n,(x,y,z) in enumerate(nodes):
            bits[z,y,x]=flags|(1<<16)|((10+n%3)<<17)
            for bank,value in enumerate([16|(16<<16),16|(1<<16),320|(640<<16),960|(0x4210<<16)]):records[z,y+bank*h,x]=value
        occupancy.write(bits.tobytes());light.write(records.tobytes());p.run(2,1,1);barrier()
        result=np.frombuffer(occupancy.read(),dtype='i4').reshape(bits.shape).copy()
        for x,y,z in nodes:assert (result[z,y,x]&~(63<<16))==(bits[z,y,x]&~(63<<16))
        return bits,result,np.frombuffer(light.read(),dtype='u4').reshape(records.shape).copy()
    nodes=[(33,4,4),(34,4,4),(35,4,4)]
    outputs=[]
    for _ in range(4):
        old,new,records=run(nodes)
        emitters=[(x,y,z) for x,y,z in nodes if new[z,y,x]&(1<<16)]
        assert len(emitters)==1
        x,y,z=emitters[0];assert new[z,y,x]>>17&31==12
        assert records[z,y+h,x]>>16==1
        outputs.append((new,records))
    for new,records in outputs[1:]:
        np.testing.assert_array_equal(new,outputs[0][0]);np.testing.assert_array_equal(records,outputs[0][1])
    big=[(x,y,5) for y in range(3,7) for x in range(3,39)][:129]
    old,new,_=run(big);np.testing.assert_array_equal(new,old)
    print('PASS actual entity clumping: deterministic cross-workgroup owner and representative, flags intact, four exact repeats; 129-node cluster retains originals')

if __name__=='__main__':
    import sys, subprocess
    cases={'rolling':rolling,'gi_diffusion':gi_diffusion,'screen_snapshots':screen_snapshots,'temporal_gating':temporal_gating,'bank_sampling':bank_sampling,'palette_extension':palette_extension,'entity_clumping':entity_clumping}
    if len(sys.argv)>1:
        print('GPU',ctx.info['GL_RENDERER'],flush=True);cases[sys.argv[1]]()
    else:
        # Fresh GL context per fixture avoids this driver's native crash when
        # sequential ModernGL tests recycle bound integer image object names.
        for name in cases:
            subprocess.run([sys.executable,'-u',__file__,name],check=True)
        print('PASS voxel history numerical regression suite')
