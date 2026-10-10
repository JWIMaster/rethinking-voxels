// GLSL port of Kino/Streak by Keijiro Takahashi (MIT).
// https://github.com/keijiro/KinoStreak -- see /licenses/KinoStreak.txt.
// Iris runs this chain before composite3: linear HDR, before world blur and
// its per-channel chromatic offsets. Composite5 only adds the finished streak.
#include "/lib/common.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(STREAK_DISPATCH_SCALE, 0.5);

#if defined KINO_STREAK && STREAK_INTENSITY > 0
layout(rgba16f) uniform image2D streakDownImage;
layout(rgba16f) uniform image2D streakUpImage;
#include "/lib/misc/kinoStreakLayout.glsl"

// Explicit horizontal interpolation keeps sampling inside one atlas tile,
// independent of Iris's custom-image sampler filtering and texture wrap mode.
vec3 StreakReadDown(float u, int y, int fullWidth, int level) {
    int width = StreakWidth(fullWidth, level);
    int offset = StreakOffset(fullWidth, level);
    float x = clamp(u * float(width) - 0.5, 0.0, float(width - 1));
    int x0 = int(floor(x));
    return mix(imageLoad(streakDownImage, ivec2(offset + x0, y)).rgb,
               imageLoad(streakDownImage, ivec2(offset + min(x0 + 1, width - 1), y)).rgb, fract(x));
}

vec3 StreakReadUp(float u, int y, int fullWidth, int level) {
    int width = StreakWidth(fullWidth, level);
    int offset = StreakOffset(fullWidth, level);
    float x = clamp(u * float(width) - 0.5, 0.0, float(width - 1));
    int x0 = int(floor(x));
    return mix(imageLoad(streakUpImage, ivec2(offset + x0, y)).rgb,
               imageLoad(streakUpImage, ivec2(offset + min(x0 + 1, width - 1), y)).rgb, fract(x));
}

#if defined STREAK_DOWN && STREAK_LEVEL == 0
vec3 StreakWorldTexel(ivec2 p) {
    // Sun/moon, clouds and the bright sky have no geometry depth. Reject
    // those source texels BEFORE interpolation so they cannot bleed into a
    // neighboring world pixel and seed a streak at the skyline.
    bool worldPixel = texelFetch(depthtex0, p, 0).r < 1.0;
    #ifdef DISTANT_HORIZONS
        ivec2 sceneSize = textureSize(colortex0, 0);
        ivec2 dhSize = textureSize(dhDepthTex, 0);
        ivec2 dhCoord = clamp(ivec2((vec2(p) + 0.5) * vec2(dhSize) / vec2(sceneSize)), ivec2(0), dhSize - 1);
        worldPixel = worldPixel || texelFetch(dhDepthTex, dhCoord, 0).r < 1.0;
    #endif
    if (!worldPixel) return vec3(0.0);
    return texelFetch(colortex0, p, 0).rgb;
}

vec3 StreakWorldVertical(vec2 uv) {
    ivec2 size = textureSize(colortex0, 0);
    // PrefilterLinear calls this only at source texel-center x coordinates.
    int x = clamp(int(uv.x * float(size.x)), 0, size.x - 1);
    float y = clamp(uv.y * float(size.y) - 0.5, 0.0, float(size.y - 1));
    int y0 = int(floor(y));
    return mix(StreakWorldTexel(ivec2(x, y0)),
               StreakWorldTexel(ivec2(x, min(y0 + 1, size.y - 1))), fract(y));
}

vec3 StreakPrefilter(vec2 uv) {
    float dy = 0.75 / float(textureSize(colortex0, 0).y);
    // Original two-tap vertical prefilter, now applied to depth-masked HDR.
    vec3 c = (StreakWorldVertical(uv - vec2(0.0, dy))
            + StreakWorldVertical(uv + vec2(0.0, dy))) * 0.5;
    float brightness = max(c.r, max(c.g, c.b));
    return c * max(0.0, brightness - STREAK_THRESHOLD) / max(brightness, 1e-5);
}

vec3 StreakPrefilterLinear(vec2 uv, int fullWidth) {
    // Reproduce linear sampling of the original thresholded full-width
    // prefilter, rather than thresholding an already interpolated source.
    float x = clamp(uv.x * float(fullWidth) - 0.5, 0.0, float(fullWidth - 1));
    float x0 = floor(x);
    vec3 a = StreakPrefilter(vec2((x0 + 0.5) / float(fullWidth), uv.y));
    vec3 b = StreakPrefilter(vec2((min(x0 + 1.0, float(fullWidth - 1)) + 0.5) / float(fullWidth), uv.y));
    return mix(a, b, fract(x));
}
#endif
#endif

void main() {
    #if defined KINO_STREAK && STREAK_INTENSITY > 0
        ivec2 sceneSize = textureSize(colortex0, 0);
        ivec2 expectedSize = ivec2(sceneSize.x, max(1, sceneSize.y / 2));
        // Refuse missing/misbound custom images instead of writing through an
        // image uniform left at its default binding (which may belong to voxel lighting).
        if (any(notEqual(imageSize(streakDownImage), expectedSize)) ||
            any(notEqual(imageSize(streakUpImage), expectedSize))) return;
        int fullWidth = expectedSize.x;
        int height = expectedSize.y;
        int lastLevel = StreakLastLevel(fullWidth);
        if (STREAK_LEVEL > lastLevel) return;
        ivec2 p = ivec2(gl_GlobalInvocationID.xy);
        int width = StreakWidth(fullWidth, STREAK_LEVEL);
        if (p.x >= width || p.y >= height) return;
        float u = (float(p.x) + 0.5) / float(width);
        vec3 c = vec3(0.0);

        #ifdef STREAK_DOWN
            #if STREAK_LEVEL == 0
                float dx = 1.25 / float(fullWidth);
                float v = (float(p.y) + 0.5) / float(height);
            #else
                float dx = 1.25 / float(StreakWidth(fullWidth, STREAK_LEVEL - 1));
            #endif
            // KinoStreak's six-tap horizontal box filter: -5,-3,-1,+1,+3,+5.
            for (int i = 0; i < 6; ++i) {
                float sampleU = u + float(i * 2 - 5) * dx;
                #if STREAK_LEVEL == 0
                    // Fuse prefilter and first downsample; the original does not
                    // reuse its full-width prefilter during reconstruction.
                    sampleU = clamp(sampleU, 0.5 / float(fullWidth), 1.0 - 0.5 / float(fullWidth));
                    c += StreakPrefilterLinear(vec2(sampleU, v), fullWidth);
                #else
                    c += StreakReadDown(sampleU, p.y, fullWidth, STREAK_LEVEL - 1);
                #endif
            }
            c /= 6.0;
            imageStore(streakDownImage, ivec2(StreakOffset(fullWidth, STREAK_LEVEL) + p.x, p.y), vec4(c, 1.0));
        #else
            c = imageLoad(streakDownImage, ivec2(StreakOffset(fullWidth, STREAK_LEVEL) + p.x, p.y)).rgb;
            if (STREAK_LEVEL < lastLevel) {
                c = mix(c, StreakReadUp(u, p.y, fullWidth, STREAK_LEVEL + 1), STREAK_STRETCH);
            }
            // Read smaller tile, write larger tile: disjoint texels within a
            // dispatch. Iris's between-dispatch barriers order the dependencies.
            imageStore(streakUpImage, ivec2(StreakOffset(fullWidth, STREAK_LEVEL) + p.x, p.y), vec4(c, 1.0));
        #endif
    #endif
}
