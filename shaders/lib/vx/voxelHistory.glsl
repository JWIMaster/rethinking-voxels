#ifndef VOXEL_HISTORY
#define VOXEL_HISTORY
ivec3 VoxelCameraDelta() {
    return cameraPositionInt.y == -98257195 ?
        ivec3(floor(cameraPosition) - floor(previousCameraPosition)) :
        cameraPositionInt - previousCameraPositionInt;
}
bool VoxelHistoryValid() {
    // A new pipeline or a jump beyond the rolling volume has no usable history.
    return frameCounter > 0 && all(lessThan(abs(VoxelCameraDelta()), voxelVolumeSize));
}
bool VoxelHistoryCoordValid(ivec3 p) {
    return all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, voxelVolumeSize));
}
#endif
