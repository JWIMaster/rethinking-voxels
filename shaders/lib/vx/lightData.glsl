#ifndef VOXEL_LIGHT_DATA
#define VOXEL_LIGHT_DATA

// Exact per-voxel records, not a lossy camera-relative hash table. Four R32UI
// planes retain integer atomics and stay below the driver's SSBO size limit.
// Iris clears this image before each frame's voxelization.
layout(r32ui) uniform uimage3D voxelLightDataI;

bool validVoxelLightCoord(ivec3 coord) {
    ivec3 expectedSize = voxelVolumeSize * ivec3(1, 4, 1);
    return all(equal(imageSize(voxelLightDataI), expectedSize)) &&
           all(greaterThanEqual(coord, ivec3(0))) && all(lessThan(coord, voxelVolumeSize));
}

ivec3 voxelLightPlaneCoord(ivec3 coord, int plane) {
    return coord + ivec3(0, plane * voxelVolumeSize.y, 0);
}

uvec4 loadVoxelLightData(ivec3 coord) {
    if (!validVoxelLightCoord(coord)) return uvec4(0u);
    uvec4 data;
    for (int k = 0; k < 4; ++k) {
        data[k] = imageLoad(voxelLightDataI, voxelLightPlaneCoord(coord, k)).r;
    }
    return data;
}

void addVoxelLightData(ivec3 coord, uvec4 data) {
    if (!validVoxelLightCoord(coord)) return;
    for (int k = 0; k < 4; ++k) {
        imageAtomicAdd(voxelLightDataI, voxelLightPlaneCoord(coord, k), data[k]);
    }
}

void storeVoxelLightData(ivec3 coord, uvec4 data) {
    if (!validVoxelLightCoord(coord)) return;
    for (int k = 0; k < 4; ++k) {
        imageAtomicExchange(voxelLightDataI, voxelLightPlaneCoord(coord, k), data[k]);
    }
}

// Preserve the legacy tag operation order: (old | orMask) & andMask.
void updateVoxelLightTag(ivec3 coord, uint andMask, uint orMask) {
    if (!validVoxelLightCoord(coord)) return;
    ivec3 tagCoord = voxelLightPlaneCoord(coord, 3);
    imageAtomicOr(voxelLightDataI, tagCoord, orMask);
    imageAtomicAnd(voxelLightDataI, tagCoord, andMask);
}
#endif
