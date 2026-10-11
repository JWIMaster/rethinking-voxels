#include "/lib/common.glsl"
#include "/lib/vx/voxelHistory.glsl"
#if VX_VOL_SIZE == 0
const ivec3 workGroups = ivec3(12, 8, 12);
#elif VX_VOL_SIZE == 1
const ivec3 workGroups = ivec3(16, 12, 16);
#elif VX_VOL_SIZE == 2
const ivec3 workGroups = ivec3(32, 16, 32);
#elif VX_VOL_SIZE == 3
const ivec3 workGroups = ivec3(64, 16, 64);
#endif
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
#ifdef SNAPSHOT_IRRADIANCE
layout(rgba16f) uniform readonly image3D irradianceCacheI;
layout(rgba16f) uniform writeonly image3D irradianceSnapshotI;
#endif
#ifdef SNAPSHOT_LIGHTS
layout(rgba16i) uniform readonly iimage3D lightStorage;
layout(rgba16i) uniform writeonly iimage3D lightStorageSnapshotI;
#endif
void main() {
    ivec3 p = ivec3(gl_GlobalInvocationID);
    if (!VoxelHistoryCoordValid(p)) return;
    #ifdef SNAPSHOT_IRRADIANCE
    for (int bank = 0; bank < 2; ++bank) {
        ivec3 q = p + ivec3(0, bank * voxelVolumeSize.y, 0);
        vec4 value = imageLoad(irradianceCacheI, q);
        #ifdef SNAPSHOT_PREVIOUS_FRAME
        if (!VoxelHistoryValid()) value = vec4(0.0);
        #endif
        if (any(isnan(value)) || any(isinf(value))) value = vec4(0.0);
        imageStore(irradianceSnapshotI, q, value);
    }
    #endif
    #ifdef SNAPSHOT_LIGHTS
    ivec4 light = imageLoad(lightStorage, p);
    #ifdef SNAPSHOT_PREVIOUS_FRAME
    if (!VoxelHistoryValid()) light = ivec4(0);
    #endif
    imageStore(lightStorageSnapshotI, p, light);
    #endif
}
