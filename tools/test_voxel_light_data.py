"""GPU regression for exact light records and camera-dependent hash collisions."""
from pathlib import Path
import numpy as np
import moderngl

ROOT = Path(__file__).resolve().parents[1]
HELPER = (ROOT / 'shaders/lib/vx/lightData.glsl').read_text()
VOLUME = (128, 96, 128)
W, H, D = VOLUME


def old_hash(p):
    a = [((int(p[k]) + 1000) * n) & 0xffffffff for k, n in enumerate((1597334673, 3812015801, 2798796415))]
    return ((a[0] ^ a[1] ^ a[2]) * 1597334673 & 0xffffffff) % (1 << 18)


def main():
    ctx = moderngl.create_standalone_context(require=430)
    a, b = (-22, 7, -7), (-21, -3, -12)
    assert old_hash(a) == old_hash(b) == 75130
    assert old_hash((a[0] - 1, a[1], a[2])) != old_hash((b[0] - 1, b[1], b[2]))
    preamble = '#version 430\nconst ivec3 voxelVolumeSize = ivec3(128,96,128);\n' + HELPER
    writer = ctx.compute_shader(preamble + '''
layout(local_size_x=64) in;
uniform int cameraShift;
void main() {
    uint i=gl_GlobalInvocationID.x;
    ivec3 a=ivec3(-22-cameraShift,7,-7)+voxelVolumeSize/2;
    ivec3 b=ivec3(-21-cameraShift,-3,-12)+voxelVolumeSize/2;
    addVoxelLightData((i%2u)==0u ? a : b, (i%2u)==0u ? uvec4(1,2,3,4) : uvec4(5,7,11,13));
    addVoxelLightData(ivec3(-1),uvec4(100u));
    addVoxelLightData(voxelVolumeSize,uvec4(100u));
}''')
    reader = ctx.compute_shader(preamble + '''
layout(local_size_x=1) in;
layout(std430,binding=0) buffer Results { uvec4 values[5]; };
uniform int cameraShift;
void main() {
    values[0]=loadVoxelLightData(ivec3(-22-cameraShift,7,-7)+voxelVolumeSize/2);
    values[1]=loadVoxelLightData(ivec3(-21-cameraShift,-3,-12)+voxelVolumeSize/2);
    values[2]=loadVoxelLightData(ivec3(-1));
    values[3]=loadVoxelLightData(voxelVolumeSize);
    values[4]=loadVoxelLightData(ivec3(0));
}''')
    tagger = ctx.compute_shader(preamble + '''
layout(local_size_x=1) in;
void main() {
    ivec3 p=ivec3(-22,7,-7)+voxelVolumeSize/2;
    storeVoxelLightData(p,uvec4(2,3,5,0x01234567u));
    updateVoxelLightTag(p,0x84a0ffffu,0xffff0000u);
}''')
    out = ctx.buffer(reserve=5*16)
    out.bind_to_storage_buffer(0)
    for shift in (-8, -1, 0, 1, 8):
        zeros = np.zeros((D,4*H,W), dtype='u4')
        atlas = ctx.texture3d((W,4*H,D),1,zeros.tobytes(),dtype='u4')
        atlas.bind_to_image(0,read=True,write=True)
        writer['voxelLightDataI'].value=0
        reader['voxelLightDataI'].value=0
        writer['cameraShift'].value=reader['cameraShift'].value=shift
        writer.run(4,1,1)
        ctx.memory_barrier()
        reader.run(1,1,1)
        ctx.memory_barrier()
        actual=np.frombuffer(out.read(),dtype='u4').reshape(5,4)
        np.testing.assert_array_equal(actual[:2],np.array([[128,256,384,512],[640,896,1408,1664]],dtype='u4'))
        np.testing.assert_array_equal(actual[2:],0)
        # Every other cell remains untouched; invalid writes never clamp/wrap.
        raw=np.frombuffer(atlas.read(),dtype='u4').reshape(D,4*H,W)
        assert np.count_nonzero(raw)==8
        if shift==0:
            tagger['voxelLightDataI'].value=0
            tagger.run(1,1,1); ctx.memory_barrier()
            reader.run(1,1,1); ctx.memory_barrier()
            actual=np.frombuffer(out.read(),dtype='u4').reshape(5,4)
            np.testing.assert_array_equal(actual[0],[2,3,5,(0x01234567|0xffff0000)&0x84a0ffff])
        atlas.release()
    # Missing/wrong atlas binding must not corrupt another volume.
    wrong=ctx.texture3d((W,H,D),1,np.full((D,H,W),91,dtype='u4').tobytes(),dtype='u4')
    wrong.bind_to_image(0,read=True,write=True)
    writer.run(4,1,1); ctx.memory_barrier()
    reader.run(1,1,1); ctx.memory_barrier()
    np.testing.assert_array_equal(np.frombuffer(out.read(),dtype='u4'),0)
    np.testing.assert_array_equal(np.frombuffer(wrong.read(),dtype='u4'),91)
    wrong.release(); out.release(); writer.release(); reader.release(); tagger.release()
    print('PASS exact voxel light records: reproduced old collision, 5 camera shifts, 256 atomic contributors, tags, bounds and wrong-size guards; '+ctx.info['GL_RENDERER'])


if __name__=='__main__':
    main()
