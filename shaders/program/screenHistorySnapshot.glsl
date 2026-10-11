#include "/lib/common.glsl"
#include "/lib/vx/voxelHistory.glsl"
#ifdef PER_PIXEL_LIGHT
const vec2 workGroupsRender = vec2(1.0, 1.0);
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f) uniform readonly image2D colorimg12;
layout(rgba16i) uniform readonly iimage2D colorimg11;
layout(rgba16f) uniform writeonly image2D diffuseHistoryI;
layout(rgba16i) uniform writeonly iimage2D candidateHistoryI;
#ifdef BLOCKLIGHT_HIGHLIGHT
layout(rgba16f) uniform readonly image2D colorimg14;
layout(rgba16f) uniform writeonly image2D specularHistoryI;
#endif
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(colorimg12)))) return;
    vec4 diffuse = imageLoad(colorimg12, p);
    if (!VoxelHistoryValid() || any(isnan(diffuse)) || any(isinf(diffuse))) diffuse = vec4(0);
    imageStore(diffuseHistoryI, p, diffuse);
    imageStore(candidateHistoryI, p, VoxelHistoryValid() ? imageLoad(colorimg11, p) : ivec4(0));
    #ifdef BLOCKLIGHT_HIGHLIGHT
    vec4 specular = imageLoad(colorimg14, p);
    if (!VoxelHistoryValid() || any(isnan(specular)) || any(isinf(specular))) specular = vec4(0);
    imageStore(specularHistoryI, p, specular);
    #endif
}
#else
const ivec3 workGroups = ivec3(1, 1, 1);
layout(local_size_x = 1) in;
void main() {}
#endif
