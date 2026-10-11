#include "/lib/common.glsl"
#include "/lib/vx/voxelHistory.glsl"

//////Fragment Shader//////Fragment Shader//////
#ifdef FRAGMENT_SHADER

in mat4 unprojectionMatrix, projectionMatrix;

vec2 view = vec2(viewWidth, viewHeight);

uniform sampler2D colortex10;

layout(rgba16f) uniform image2D colorimg8;
#ifdef BLOCKLIGHT_HIGHLIGHT
    layout(rgba8) uniform image2D colorimg3;
#endif
layout(r32ui) uniform restrict readonly uimage2D colorimg9;

#define MATERIALMAP_ONLY
#include "/lib/vx/SSBOs.glsl"

void main() {
    if (!VoxelHistoryValid()) discard;
    ivec2 p = ivec2(gl_FragCoord.xy);
    uint encodedDepth = imageLoad(colorimg9, p).r;
    if (encodedDepth >= (1u << 30)) discard;
    float newDepth = float(encodedDepth) / float(1u << 30);
    vec4 oldClip = vec4(gl_FragCoord.xy / view, newDepth, 1);
    if (newDepth > 0.56) {
        vec4 currentClip = oldClip * 2 - 1;
        currentClip.w = 1.0;
        vec4 player = gbufferModelViewInverse * (gbufferProjectionInverse * currentClip);
        if (player.w <= 1e-8 || any(isnan(player)) || any(isinf(player))) discard;
        player.xyz += player.w * (cameraPosition - previousCameraPosition);
        oldClip = gbufferPreviousProjection * (gbufferPreviousModelView * player);
        if (oldClip.w <= 1e-8 || any(isnan(oldClip)) || any(isinf(oldClip))) discard;
        oldClip = oldClip / oldClip.w * 0.5 + 0.5;
    } // Hand/viewmodel depth deliberately stays in the same screen pixels.
    vec2 halfTexel = 0.5 / view;
    if (any(lessThan(oldClip.xy, halfTexel)) || any(greaterThan(oldClip.xy, 1 - halfTexel))) discard;
    // Each destination has one writer even when many old pixels shared the
    // same quantized nearest depth. No equality-tie scatter race remains.
    vec3 normal = newDepth < 0.999998 ? texture(colortex4, oldClip.xy).gba * 2 - 1 : vec3(0);
    if (any(isnan(normal)) || any(isinf(normal))) discard;
    imageStore(colorimg8, p, vec4(normal, 1 - newDepth));
    #ifdef BLOCKLIGHT_HIGHLIGHT
        imageStore(colorimg3, p, texture(colortex6, oldClip.xy));
    #endif
    /*DRAWBUFFERS:3*/
    discard;
}
#endif

//////Vertex Shader//////Vertex Shader//////
#ifdef VERTEX_SHADER

out mat4 unprojectionMatrix, projectionMatrix;

#define MATERIALMAP_ONLY
#include "/lib/vx/SSBOs.glsl"

void main() {
    projectionMatrix =
        gbufferProjection *
        gbufferModelView;
    unprojectionMatrix =
        gbufferPreviousModelViewInverse * 
        gbufferPreviousProjectionInverse;
    gl_Position = ftransform();
}
#endif