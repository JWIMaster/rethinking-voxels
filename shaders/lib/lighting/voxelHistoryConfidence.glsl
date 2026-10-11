#ifndef VOXEL_HISTORY_CONFIDENCE
#define VOXEL_HISTORY_CONFIDENCE

float voxelViewDepth(float rawDepth, float nearPlane, float farPlane) {
    if (isnan(rawDepth) || isinf(rawDepth) || rawDepth < 0.0 || rawDepth >= 1.0 ||
        isnan(nearPlane) || isinf(nearPlane) || isnan(farPlane) || isinf(farPlane) ||
        nearPlane <= 0.0 || farPlane <= nearPlane) return -1.0;
    return nearPlane / max(1.0 - rawDepth + (nearPlane / farPlane) * rawDepth, 1e-8);
}

float voxelHistoryConfidence(vec3 currentNormal, float currentRawDepth,
                             vec4 savedNormalInverseDepth, float nearPlane, float farPlane) {
    if (any(isnan(currentNormal)) || any(isinf(currentNormal)) ||
        any(isnan(savedNormalInverseDepth)) || any(isinf(savedNormalInverseDepth)) ||
        savedNormalInverseDepth.w <= 0.0 || savedNormalInverseDepth.w > 1.0) return 0.0;
    float currentLength = length(currentNormal);
    float savedLength = length(savedNormalInverseDepth.xyz);
    if (currentLength < 0.1 || savedLength < 0.1 ||
        isnan(currentLength) || isinf(currentLength) || isnan(savedLength) || isinf(savedLength)) return 0.0;
    float currentDepth = voxelViewDepth(currentRawDepth, nearPlane, farPlane);
    float savedDepth = voxelViewDepth(1.0 - savedNormalInverseDepth.w, nearPlane, farPlane);
    if (currentDepth < 0.0 || savedDepth < 0.0) return 0.0;
    float depthTolerance = max(0.1, 0.02 * currentDepth);
    float depthError = abs(savedDepth - currentDepth) / depthTolerance;
    float normalError = length(currentNormal / currentLength - savedNormalInverseDepth.xyz / savedLength) / 0.3;
    // Exact and close same-surface matches retain full quality. Approaching
    // either rejection limit fades continuously into the volume cache.
    return (1.0 - smoothstep(0.25, 1.0, depthError)) *
           (1.0 - smoothstep(0.25, 1.0, normalError));
}

float voxelHistorySpatialWeight(vec2 pixelOffset) {
    return 1.0 / (1.0 + 0.1 * dot(pixelOffset, pixelOffset));
}

#endif
