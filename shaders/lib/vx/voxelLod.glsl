#ifndef VOXEL_LOD_HELPERS
#define VOXEL_LOD_HELPERS

bool voxelCoordInBounds(ivec3 coord) {
    return all(greaterThanEqual(coord, ivec3(0))) && all(lessThan(coord, voxelVolumeSize));
}

ivec3 voxelCellCoord(vec3 pos) {
    if (any(isnan(pos)) || any(isinf(pos))) return ivec3(-1);
    // floor is required on the negative edge; an int cast truncates toward zero.
    return ivec3(floor(pos + 0.5 * vec3(voxelVolumeSize)));
}

bool voxelPositionInBounds(vec3 pos) {
    return all(greaterThanEqual(pos, -0.5 * vec3(voxelVolumeSize))) &&
           all(lessThan(pos, 0.5 * vec3(voxelVolumeSize)));
}

int getVoxelWriteLevelCount(vec3 minAbsPos) {
    if (any(isnan(minAbsPos)) || any(isinf(minAbsPos))) return 0;
    float radius = infnorm(minAbsPos / vec3(voxelVolumeSize));
    int count = 0;
    // Power-of-two comparisons avoid log(0), undefined float-to-int conversion
    // and a driver's approximate log2 rounding just below integer boundaries.
    for (int level = 0; level < VOXEL_DETAIL_AMOUNT; ++level) {
        if (radius > exp2(-float(level + 1))) break;
        count = level + 1;
    }
    return count;
}

int getVoxelResolution(vec3 pos) {
    if (any(isnan(pos)) || any(isinf(pos))) return 0;
    vec3 safeSize = max(vec3(voxelVolumeSize) - 2.01, vec3(1.0));
    float radius = infnorm(pos / safeSize);
    int resolution = 0;
    for (int level = 1; level < VOXEL_DETAIL_AMOUNT; ++level) {
        if (radius > exp2(-float(level + 1))) break;
        resolution = level;
    }
    return resolution;
}

#endif
