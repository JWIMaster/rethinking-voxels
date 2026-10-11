"""GPU checks of actual consumer confidence and extracted main-lighting paths."""
from pathlib import Path
import numpy as np
import moderngl

ROOT=Path(__file__).resolve().parents[1]/'shaders'

def raw_depth(distance,n=.05,f=320.):
    return np.float32((1-n/distance)/(1-n/f))

def main():
    ctx=moderngl.create_standalone_context(require=430)
    print(ctx.info['GL_RENDERER'],ctx.info['GL_VERSION'])
    helper=(ROOT/'lib/lighting/voxelHistoryConfidence.glsl').read_text()
    source='#version 430\n'+helper+'''
layout(local_size_x=1) in;
layout(std430,binding=0) readonly buffer In { vec4 inputData[]; };
layout(std430,binding=1) buffer Out { vec4 result[]; };
void main(){uint i=gl_GlobalInvocationID.x;vec4 a=inputData[3*i],b=inputData[3*i+1],c=inputData[3*i+2];result[i]=vec4(voxelHistoryConfidence(a.xyz,a.w,b,c.x,c.y),voxelViewDepth(a.w,c.x,c.y),voxelHistorySpatialWeight(c.zw),1);}
'''
    shader=ctx.compute_shader(source)
    cases=[]
    def add(current,saved,current_normal=(0,1,0),saved_normal=(0,1,0),n=.05,f=320,offset=(0,0)):
        cases.append([[*current_normal,raw_depth(current,n,f)],[*saved_normal,np.float32(1-raw_depth(saved,n,f))],[n,f,*offset]])
    add(10,200)
    for d in [.1,1,10,50,200,319]: add(d,d)
    start=len(cases)
    for d in np.linspace(9.75,10.25,101): add(10,d)
    end=len(cases)
    move_start=len(cases)
    for camera_delta in np.linspace(0,.25,51): add(10-camera_delta,10)
    move_end=len(cases)
    normal_start=len(cases)
    for angle in np.linspace(0,.4,81): add(10,10,saved_normal=(np.sin(angle),np.cos(angle),0))
    normal_end=len(cases)
    bad_start=len(cases)
    for target,component,value in [(0,0,np.nan),(0,1,np.inf),(0,3,np.nan),(0,3,-.1),(0,3,1),(1,0,np.nan),(1,1,np.inf),(1,3,0),(1,3,2),(2,0,0),(2,0,np.nan),(2,1,np.inf),(2,1,.001)]:
        add(10,10);cases[-1][target][component]=value
    add(10,10,current_normal=(0,0,0));add(10,10,saved_normal=(0,0,0))
    add(10,10,current_normal=(3e38,0,0));add(10,10,saved_normal=(3e38,0,0))
    inputs=np.asarray(cases,dtype='f4')
    ib=ctx.buffer(inputs.tobytes());ob=ctx.buffer(reserve=len(cases)*16)
    ib.bind_to_storage_buffer(0);ob.bind_to_storage_buffer(1)
    shader.run(group_x=len(cases));ctx.memory_barrier()
    out=np.frombuffer(ob.read(),dtype='f4').reshape(-1,4)
    assert np.isfinite(out).all()
    assert out[0,0]==0
    old_error=60*abs(float(inputs[0,0,3])-float(1-inputs[0,1,3]))
    assert old_error<.3,old_error
    np.testing.assert_array_equal(out[1:7,0],1)
    profile=out[start:end,0]
    assert profile[50]==1 and profile[0]==profile[-1]==0
    assert (np.diff(profile[:51])>=-1e-4).all() and (np.diff(profile[50:])<=1e-4).all()
    assert np.max(np.abs(np.diff(profile)))<.06
    movement=out[move_start:move_end,0]
    assert movement[0]==1 and movement[-1]==0 and (np.diff(movement)<=1e-4).all()
    assert np.max(np.abs(np.diff(movement)))<.07
    normals=out[normal_start:normal_end,0]
    assert normals[0]==1 and normals[-1]==0 and (np.diff(normals)<=1e-4).all()
    np.testing.assert_array_equal(out[bad_start:,0],0)
    print('PASS',len(cases),'actual confidence cases: old10m/200m false match rejected, same-surface1, smooth depth/normal/motion, invalid inputs0')
    shader.release();ib.release();ob.release()

    main_source=(ROOT/'lib/lighting/mainLighting.glsl').read_text()
    diffuse=main_source[main_source.index('    // Voxel-based Lighting'):main_source.index('    #ifdef GI',main_source.index('    // Voxel-based Lighting'))]
    spec_start=main_source.index('    #if defined BLOCKLIGHT_HIGHLIGHT && defined PER_PIXEL_LIGHT && !defined GBUFFERS_WATER',main_source.index('    // Light Highlight'))
    spec=main_source[spec_start:main_source.index('        #ifdef CUSTOM_PBR',spec_start)]+'\n#endif\n'
    vertex='''#version 330
uniform float rawDepth;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2-1,rawDepth*2-1,1);}
'''
    cache=np.array([.4,.8,1.2],dtype='f4');pixel=np.array([2,3,4],dtype='f4');highlight=np.array([8,12,16],dtype='f4')
    cases_rendered=0
    for mode in ['ppl','water','no-ppl','hand']:
        flags='#define PER_PIXEL_LIGHT\n#define BLOCKLIGHT_HIGHLIGHT\n' if mode!='no-ppl' else ''
        if mode=='water': flags+='#define GBUFFERS_WATER\n'
        if mode=='hand': flags+='#define GBUFFERS_HAND\n'
        frag='#version 330\n'+flags+helper+'''
uniform sampler2D colortex8,colortex12,colortex14;
uniform float near,far;
uniform vec3 worldNormalM;
vec3 vxPos=vec3(0);
float nextFloat(){return .321;}
vec3 readSurfaceVoxelBlocklight(vec3 p,vec3 n){return vec3(.1,.2,.3);}
layout(location=0) out vec4 outputDiffuse;
layout(location=1) out vec4 outputSpecular;
void main(){
'''+diffuse+spec+'''
outputDiffuse=vec4(voxelBlockLighting,1);
#if defined BLOCKLIGHT_HIGHLIGHT && defined PER_PIXEL_LIGHT && !defined GBUFFERS_WATER
outputSpecular=vec4(blocklightHighlight,1);
#else
outputSpecular=vec4(0,0,0,1);
#endif
}
'''
        program=ctx.program(vertex_shader=vertex,fragment_shader=frag)
        vao=ctx.vertex_array(program,[])
        for distance in [10,10.05,10.1,10.19,10.25,200,np.nan]:
            nd=np.zeros((4,4,4),dtype='f4');nd[:,:,1]=1
            nd[:,:,3]=np.float32(1-raw_depth(distance)) if np.isfinite(distance) else np.nan
            textures=[ctx.texture((4,4),4,nd.tobytes(),dtype='f4'),ctx.texture((4,4),3,np.broadcast_to(pixel/4,(4,4,3)).copy().tobytes(),dtype='f4'),ctx.texture((4,4),3,np.broadcast_to(highlight/4,(4,4,3)).copy().tobytes(),dtype='f4')]
            for i,(name,tex) in enumerate(zip(['colortex8','colortex12','colortex14'],textures)):
                tex.use(i)
                if name in program:program[name].value=i
            for name,value in [('near',.05),('far',320.),('worldNormalM',(0,1,0)),('rawDepth',float(raw_depth(10)))]:
                if name in program:program[name].value=value
            targets=[ctx.texture((4,4),4,dtype='f4'),ctx.texture((4,4),4,dtype='f4')]
            fbo=ctx.framebuffer(color_attachments=targets);fbo.use();ctx.viewport=(0,0,4,4)
            vao.render(mode=moderngl.TRIANGLES,vertices=3)
            actual=[np.frombuffer(t.read(),dtype='f4').reshape(4,4,4)[:,:,:3] for t in targets]
            assert all(np.isfinite(a).all() for a in actual)
            if mode in ['water','no-ppl']:
                np.testing.assert_allclose(actual[0],np.broadcast_to(cache,actual[0].shape),atol=2e-6)
                np.testing.assert_array_equal(actual[1],0)
            elif distance==10:
                np.testing.assert_array_equal(actual[0],np.broadcast_to(pixel,actual[0].shape))
                np.testing.assert_array_equal(actual[1],np.broadcast_to(highlight,actual[1].shape))
            elif distance>=10.25 or not np.isfinite(distance):
                np.testing.assert_allclose(actual[0],np.broadcast_to(cache,actual[0].shape),atol=2e-6)
                np.testing.assert_array_equal(actual[1],0)
            else:
                normalized=(actual[0][0,0]-cache)/(pixel-cache)
                np.testing.assert_allclose(normalized,normalized[0],atol=2e-6)
                assert 0<normalized[0]<=1
                np.testing.assert_allclose(actual[1][0,0]/highlight,normalized[0],atol=2e-6)
            cases_rendered+=1
            fbo.release()
            for t in targets+textures:t.release()
        vao.release();program.release()
    assert 'foundValidLoc = true' not in main_source
    assert 'confidence *= voxelHistorySpatialWeight(pixelOffset)' in main_source
    assert 'pixelWeightSum' in main_source
    print('PASS',cases_rendered,'actual consumer renders: center exact, confidence mixture, specular same fade, hand rejects invalid, water/noPPL unchanged, small-screen edges bounded')
    print('ALL VOXEL CONFIDENCE REGRESSIONS PASS')

if __name__=='__main__':main()
