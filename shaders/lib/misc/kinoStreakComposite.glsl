// Kino/Streak final composition in linear HDR, before color grading / tone mapping.
uniform sampler2D streakUpSampler;

vec3 GetKinoStreak(vec2 uv, ivec2 sceneSize) {
    ivec2 atlasSize = textureSize(streakUpSampler, 0);
    // An unregistered sampler defaults to unit 0 (colortex0 in Iris). Never
    // treat that full-height scene texture as our half-height streak atlas.
    ivec2 expectedSize = ivec2(sceneSize.x, max(1, sceneSize.y / 2));
    if (any(notEqual(atlasSize, expectedSize))) return vec3(0.0);
    ivec2 size = ivec2(max(1, atlasSize.x / 2), atlasSize.y);
    vec2 p = clamp(uv * vec2(size) - 0.5, vec2(0.0), vec2(size - 1));
    ivec2 p0 = ivec2(floor(p));
    ivec2 p1 = min(p0 + 1, size - 1);
    vec2 f = fract(p);
    // Manual bilinear sampling avoids neighboring atlas levels and supports
    // custom images regardless of the sampler's filtering configuration.
    vec3 a = mix(texelFetch(streakUpSampler, p0, 0).rgb,
                 texelFetch(streakUpSampler, ivec2(p1.x, p0.y), 0).rgb, f.x);
    vec3 b = mix(texelFetch(streakUpSampler, ivec2(p0.x, p1.y), 0).rgb,
                 texelFetch(streakUpSampler, p1, 0).rgb, f.x);
    vec3 streak = mix(a, b, f.y);
    #if STREAK_COLOR_MODE == 1
        // Classic blue lens coating: preserve highlight strength, not its hue.
        // This also turns warm/red sources blue, unlike multiplying RGB by a tint.
        float brightness = max(0.0, max(streak.r, max(streak.g, streak.b)));
        streak = brightness * vec3(0.25, 0.55, 1.0);
    #else
        // Original appearance, including the user's existing tint sliders.
        streak *= vec3(STREAK_TINT_R, STREAK_TINT_G, STREAK_TINT_B);
    #endif
    return streak * (float(STREAK_INTENSITY) * 0.05);
}
