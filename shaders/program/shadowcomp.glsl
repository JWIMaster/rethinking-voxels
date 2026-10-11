#include "/lib/common.glsl"
#include "/lib/vx/voxelHistory.glsl"
#include "/lib/vx/lightData.glsl"

#ifdef CSH
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

layout(rgba16f) uniform image3D distanceFieldI;
layout(r32i) uniform restrict iimage3D occupancyVolume;
layout(r32i) uniform restrict iimage3D voxelCols;

ivec3 floorCamPosOffset =
    cameraPositionInt.y == -98257195 ?
    ivec3((floor(cameraPosition) - floor(previousCameraPosition)) * 1.001) :
    cameraPositionInt - previousCameraPositionInt;

bvec2 or(bvec2 a, bvec2 b) {
    return bvec2(a.x || b.x, a.y || b.y);
}

bvec3 or(bvec3 a, bvec3 b) {
    return bvec3(a.x || b.x, a.y || b.y, a.z || b.z);
}

bvec4 or(bvec4 a, bvec4 b) {
    return bvec4(a.x || b.x, a.y || b.y, a.z || b.z, a.w || b.w);
}

shared float fullDist[10][10][10];
shared uint isActive;

#include "/lib/vx/positionHashing.glsl"


const float levelFadeDist = 2.0;
float ReadPreviousDistance(ivec3 coord, int level) {
    if (!VoxelHistoryValid() || !VoxelHistoryCoordValid(coord)) return 0.0;
    ivec3 bank = ivec3(0, (frameCounter % 2 * 2 + level / 4) * voxelVolumeSize.y, 0);
    float distance = imageLoad(distanceFieldI, coord + bank)[level % 4];
    return isnan(distance) || isinf(distance) ? 0.0 : distance;
}

void main() {
    ivec3 baseCoord = ivec3(gl_WorkGroupID) * 8;
    ivec3 localCoord = ivec3(gl_LocalInvocationID);
    ivec3 texCoord = baseCoord + localCoord;
    float[8] theseDists;
    for (int k = 0; k < 8; k++) theseDists[k] = 1000;
    int thisOccupancy = imageLoad(occupancyVolume, texCoord).r;
    #define j 0
    #include "/program/shadowcomp_sdf_loop.glsl"
    #undef j
    #if VOXEL_DETAIL_AMOUNT > 1
        #define j 1
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 2
        #define j 2
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 3
        #define j 3
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 4
        #define j 4
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 5
        #define j 5
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 6
        #define j 6
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 7
        #define j 7
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    imageStore(
        distanceFieldI,
        texCoord + ivec3(0, (frameCounter+1)%2 * 2 * voxelVolumeSize.y, 0),
        vec4(theseDists[0], theseDists[1], theseDists[2], theseDists[3]));
    imageStore(
        distanceFieldI,
        texCoord + ivec3(0, ((frameCounter+1)%2 * 2 + 1) * voxelVolumeSize.y, 0),
        vec4(theseDists[4], theseDists[5], theseDists[6], theseDists[7]));

    // extend colour data into adjacent air for less artefacts in corners etc
    ivec2 rawCol = ivec2(
        imageLoad(voxelCols, texCoord * ivec3(1, 2, 1)).r,
        imageLoad(voxelCols, texCoord * ivec3(1, 2, 1) + ivec3(0, 1, 0)).r
    );
    int dataBits = rawCol.x >> 26 << 26;
    if ((rawCol.r & (1 << 28)) == 0 && (rawCol.g >> 23) == 0) {
        for (int k = 0; k < 6; k++) {
            ivec3 offset = (k/3*2-1) * ivec3(equal(ivec3(k%3), ivec3(0, 1, 2)));
            if (!VoxelHistoryCoordValid(texCoord + offset)) continue;
            ivec2 otherRawCol = ivec2(
                imageLoad(voxelCols, (texCoord + offset) * ivec3(1, 2, 1)).r,
                imageLoad(voxelCols, (texCoord + offset) * ivec3(1, 2, 1) + ivec3(0, 1, 0)).r
            );
            // Only original geometry can donate: previously extended air must
            // not recursively propagate in an execution-order-dependent way.
            if ((otherRawCol.r & (1 << 28)) != 0 && (otherRawCol.g >> 23) > (rawCol.g >> 23)) {
                rawCol = otherRawCol;
                rawCol.g &= ~(0x3ff << 13);
            }
        }
        if ((rawCol.g >> 23) > 0) {
            rawCol.r &= (1<<26) - 1;
            rawCol.r |= dataBits;
            imageStore(voxelCols, texCoord * ivec3(1, 2, 1), ivec4(rawCol.r));
            imageStore(voxelCols, texCoord * ivec3(1, 2, 1) + ivec3(0, 1, 0), ivec4(rawCol.g));
        }
    }
    if ((thisOccupancy >> 16 & 1) != 0) {
        updateVoxelLightTag(texCoord, (uint(16 + (16 << 5) + (16 << 10)) << 16) | 0xffffu, 0xffff0000u);
    }
}
#endif

// entity light clumping
#ifdef CSH_A
#if VX_VOL_SIZE == 0
    const ivec3 workGroups = ivec3(3, 2, 3);
#elif VX_VOL_SIZE == 1
    const ivec3 workGroups = ivec3(4, 3, 4);
#elif VX_VOL_SIZE == 2
    const ivec3 workGroups = ivec3(8, 4, 8);
#elif VX_VOL_SIZE == 3
    const ivec3 workGroups = ivec3(16, 4, 16);
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(r32i) uniform restrict iimage3D occupancyVolume;

#include "/lib/vx/positionHashing.glsl"

// One globally defined owner per complete component. No thread clears
// membership while another discovers it; oversized components remain original.
uint EntityCoordKey(ivec3 p) {
    return uint(p.x + voxelVolumeSize.x * (p.y + voxelVolumeSize.y * p.z));
}
void ClumpEntityComponent(ivec3 source) {
    if (!VoxelHistoryCoordValid(source)) return;
    int sourceBits = imageLoad(occupancyVolume, source).r;
    if ((sourceBits & (1 << 27)) == 0 || (sourceBits & (1 << 16)) == 0) return;
    int lightClass = sourceBits >> 22 & 63;
    uint ownerKey = EntityCoordKey(source);
    // Most non-root voxels can be rejected without a component traversal.
    for (int n = 0; n < 6; ++n) {
        ivec3 d = (n / 3 * 2 - 1) * ivec3(n % 3 == 0, n % 3 == 1, n % 3 == 2);
        ivec3 q = source + d;
        if (VoxelHistoryCoordValid(q) && (imageLoad(occupancyVolume, q).r >> 22 & 63) == lightClass && EntityCoordKey(q) < ownerKey) return;
    }
    const int MAX_CLUMP_SIZE = 128;
    ivec3 nodes[MAX_CLUMP_SIZE];
    nodes[0] = source;
    int count = 1;
    bool complete = true;
    int maxLightLevel = sourceBits >> 17 & 31;
    for (int j = 0; j < count; ++j) {
        for (int n = 0; n < 6; ++n) {
            ivec3 d = (n / 3 * 2 - 1) * ivec3(n % 3 == 0, n % 3 == 1, n % 3 == 2);
            ivec3 q = nodes[j] + d;
            if (!VoxelHistoryCoordValid(q)) continue;
            int bits = imageLoad(occupancyVolume, q).r;
            if ((bits >> 22 & 63) != lightClass) continue;
            bool known = false;
            for (int k = 0; k < count; ++k) if (nodes[k] == q) { known = true; break; }
            if (known) continue;
            if (count == MAX_CLUMP_SIZE) { complete = false; continue; }
            nodes[count++] = q;
            ownerKey = min(ownerKey, EntityCoordKey(q));
            maxLightLevel = max(maxLightLevel, bits >> 17 & 31);
        }
    }
    if (!complete || ownerKey != EntityCoordKey(source)) return;
    ivec4 meanPos = ivec4(0);
    ivec3 meanCol = ivec3(0);
    for (int k = 0; k < count; ++k) {
        uvec4 data = loadVoxelLightData(nodes[k]);
        ivec4 pos = ivec4(data.x & 0xffffu, data.x >> 16, data.y & 0xffffu, data.y >> 16);
        if (pos.w == 0) return;
        meanPos += pos + ivec4(nodes[k] - source, 0) * 32 * pos.w;
        meanCol += ivec3(data.z & 0xffffu, data.z >> 16, data.w & 0xffffu);
    }
    if (meanPos.w <= 0) return;
    meanCol /= meanPos.w;
    meanPos /= meanPos.w;
    vec3 probePos = source + meanPos.xyz / 32.0 - vec3(0.991775521, 1.0061213, 1.000062142);
    float minLen = 1000000.0;
    int bestFit = 0;
    for (int k = 0; k < count; ++k) {
        float distance = length(probePos - nodes[k]);
        if (distance < minLen || (distance == minLen && EntityCoordKey(nodes[k]) < EntityCoordKey(nodes[bestFit]))) {
            minLen = distance; bestFit = k;
        }
    }
    ivec3 outputCoord = nodes[bestFit];
    meanPos.xyz += 32 * (source - outputCoord);
    uvec4 outputData = uvec4(uint(meanPos.x) | uint(meanPos.y) << 16,
        uint(meanPos.z) | 1u << 16, uint(meanCol.x) | uint(meanCol.y) << 16,
        uint(meanCol.z) | 0xffff0000u);
    for (int k = 0; k < count; ++k) imageAtomicAnd(occupancyVolume, nodes[k], ~(1 << 16));
    // Preserve geometry, translucency, material, entity and sky flags.
    imageAtomicAnd(occupancyVolume, outputCoord, ~(63 << 16));
    imageAtomicOr(occupancyVolume, outputCoord, (1 + (maxLightLevel << 1)) << 16);
    storeVoxelLightData(outputCoord, outputData);
}
void main() {
    for (int k = 0; k < 64; ++k) {
        ivec3 offset = ivec3(k % 4, k / 4 % 4, k / 16 % 4);
        ClumpEntityComponent(ivec3(gl_GlobalInvocationID) * 4 + 3 + offset);
    }
}
#endif


#ifdef CSH_B
#ifdef LIGHT_CLUMPING
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

layout(r32i) uniform restrict iimage3D occupancyVolume;

uniform vec3 cameraPosition;

#include "/lib/vx/positionHashing.glsl"

shared ivec3 lightCs[512];
shared int lightCount;
shared ivec4 lightLocs[8][8][8];
shared ivec3 lightCols[8][8][8];
void main() {
    int index = int(gl_LocalInvocationID.x) + 8 * int(gl_LocalInvocationID.y) + 64 * int(gl_LocalInvocationID.z);
    if (index == 0) {
        lightCount = 0;
    }
    ivec3 c = ivec3(gl_LocalInvocationID);
    lightLocs[c.x][c.y][c.z] = ivec4(0);
    lightCols[c.x][c.y][c.z] = ivec3(0);
    barrier();
    memoryBarrierShared();
    ivec3 coord = ivec3(gl_GlobalInvocationID) + ((cameraPositionInt.y == -98257195 ? ivec3(floor(cameraPosition)) : cameraPositionInt) % 2 + 2) % 2;
    int thisOccupancyData0 = imageLoad(occupancyVolume, coord).r;
    if ((thisOccupancyData0 >> 16 & 1) != 0 && (thisOccupancyData0 >> 27 & 1) == 0) {
        lightCs[atomicAdd(lightCount, 1)] = c;
    }
    barrier();
    memoryBarrierShared();

    bool spreadDown = false;
    if (index < lightCount) {
        c = lightCs[index];
        coord = c + 8 * ivec3(gl_WorkGroupID) + ((cameraPositionInt.y == -98257195 ? ivec3(floor(cameraPosition)) : cameraPositionInt) % 2 + 2) % 2;
        ivec3 lightLoc0 = coord - voxelVolumeSize/2;
        int thisOccupancy = imageLoad(occupancyVolume, coord).r;
        uvec4 lightData = loadVoxelLightData(coord);
        uvec2 packedLightSubPos = lightData.xy;
        uvec2 packedLightCol = lightData.zw;
        ivec4 subPos = ivec4(packedLightSubPos.x & 0xffffu, packedLightSubPos.x >> 16, packedLightSubPos.y & 0xffffu, packedLightSubPos.y >> 16);
        ivec4 col = ivec4(packedLightCol.x & 0xffffu, packedLightCol.x >> 16, packedLightCol.y & 0xffffu, packedLightCol.y >> 16);
        ivec3 downRange = coord - (coord + ((cameraPositionInt.y == -98257195 ? ivec3(floor(cameraPosition)) : cameraPositionInt) % 2 + 2) % 2)%2;
        for (int k = 0; k < 4; k++) {
            atomicAdd(lightLocs[c.x][c.y][c.z][k], subPos[k]);
        }
        for (int k = 0; k < 3; k++) {
            atomicAdd(lightCols[c.x][c.y][c.z][k], col[k]);
        }
        for (int dir1 = 0; dir1 < 3; dir1++) {
            ivec4 d1 = ivec4(0);
            ivec3 offsetCoord1 = coord;
            offsetCoord1[dir1]--;
            d1[dir1]--;
            int otherOccupancy1 = imageLoad(occupancyVolume, offsetCoord1).r;
            if (all(greaterThanEqual(offsetCoord1, downRange)) && (otherOccupancy1 >> 16 & 1) != 0 && (otherOccupancy1 >> 17 & 1023) == (thisOccupancy >> 17 & 1023)) {
                spreadDown = true;
                for (int k = 0; k < 4; k++) {
                    atomicAdd(lightLocs[c.x+d1.x][c.y+d1.y][c.z+d1.z][k], subPos[k] - 32 * d1[k] * subPos[3]);
                }
                for (int k = 0; k < 3; k++) {
                    atomicAdd(lightCols[c.x+d1.x][c.y+d1.y][c.z+d1.z][k], col[k]);
                }
                for (int dir2 = 0; dir2 < 3; dir2++) {
                    ivec4 d2 = d1;
                    ivec3 offsetCoord2 = offsetCoord1;
                    offsetCoord2[dir2]--;
                    d2[dir2]--;
                    int otherOccupancy2 = imageLoad(occupancyVolume, offsetCoord2).r;
                    if (all(greaterThanEqual(offsetCoord2, downRange)) && (otherOccupancy2 >> 16 & 1) != 0 && (otherOccupancy2 >> 17 & 1023) == (thisOccupancy >> 17 & 1023)) {
                        for (int k = 0; k < 4; k++) {
                            atomicAdd(lightLocs[c.x+d2.x][c.y+d2.y][c.z+d2.z][k], subPos[k] - 32 * d2[k] * subPos[3]);
                        }
                        for (int k = 0; k < 3; k++) {
                            atomicAdd(lightCols[c.x+d2.x][c.y+d2.y][c.z+d2.z][k], col[k]);
                        }
                        for (int dir3 = 0; dir3 < 3; dir3++) {
                            ivec4 d3 = d2;
                            ivec3 offsetCoord3 = offsetCoord2;
                            offsetCoord3[dir3]--;
                            d3[dir3]--;
                            int otherOccupancy3 = imageLoad(occupancyVolume, offsetCoord3).r;
                            if (all(greaterThanEqual(offsetCoord3, downRange)) && (otherOccupancy3 >> 16 & 1) != 0 && (otherOccupancy3 >> 17 & 1023) == (thisOccupancy >> 17 & 1023)) {
                                for (int k = 0; k < 4; k++) {
                                    atomicAdd(lightLocs[c.x+d3.x][c.y+d3.y][c.z+d3.z][k], subPos[k] - 32 * d3[k] * subPos[3]);
                                }
                                for (int k = 0; k < 3; k++) {
                                    atomicAdd(lightCols[c.x+d3.x][c.y+d3.y][c.z+d3.z][k], col[k]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    barrier();
    memoryBarrierShared();
    if (index < lightCount) {
        if (spreadDown) {
            imageAtomicAnd(occupancyVolume, coord, ~(1<<16));
        } else {
            //lightCols[c.x][c.y][c.z] /= max(1, lightLocs[c.x][c.y][c.z].w);
            //lightLocs[c.x][c.y][c.z] /= max(1, lightLocs[c.x][c.y][c.z].w);
            uvec4 packedLightData = uvec4(
                uint(lightLocs[c.x][c.y][c.z].x) | uint(lightLocs[c.x][c.y][c.z].y) << 16,
                uint(lightLocs[c.x][c.y][c.z].z) | uint(lightLocs[c.x][c.y][c.z].w) << 16,
                uint(lightCols[c.x][c.y][c.z].x) | uint(lightCols[c.x][c.y][c.z].y) << 16,
                uint(lightCols[c.x][c.y][c.z].z)
            );
            storeVoxelLightData(coord, packedLightData);
        }
    }
}
#else
const ivec3 workGroups = ivec3(1, 1, 1);
layout(local_size_x = 1) in;
void main() {}
#endif
#endif
