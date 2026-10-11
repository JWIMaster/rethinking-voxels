#ifndef IRRADIANCECACHE
#define IRRADIANCECACHE
bool isInRange(vec3 vxPos) {
    return all(greaterThan(vxPos, -0.5*voxelVolumeSize)) && all(lessThan(vxPos, 0.5*voxelVolumeSize));
}
#ifdef IRRADIANCE_SNAPSHOT_READER
uniform sampler3D irradianceSnapshot;
#define IRRADIANCE_TEXTURE irradianceSnapshot
#else
uniform sampler3D irradianceCache;
#define IRRADIANCE_TEXTURE irradianceCache
#endif
vec4 ReadIrradianceBank(vec3 pos, int bank) {
    // Clamp inside this bank's texel centers after normal bias. Filtering must
    // never cross the GI/direct-light seam in the packed two-height texture.
    vec3 p = clamp(pos + 0.5 * voxelVolumeSize, vec3(0.5), vec3(voxelVolumeSize) - 0.5);
    p.y += float(bank * voxelVolumeSize.y);
    vec3 uv = p / (vec3(voxelVolumeSize) * vec3(1.0, 2.0, 1.0));
    vec4 color = textureLod(IRRADIANCE_TEXTURE, uv, 0);
    return any(isnan(color)) || any(isinf(color)) ? vec4(0) : color;
}
vec3 readIrradianceCache(vec3 vxPos, vec3 normal) {
    if (!isInRange(vxPos)) return vec3(0);
    vec4 color = ReadIrradianceBank(vxPos + 0.5 * normal, 0);
    return color.rgb / max(color.a, 0.0001);
}
vec3 readSurfaceVoxelBlocklight(vec3 vxPos, vec3 normal) {
    if (!isInRange(vxPos)) return vec3(0);
    vec3 color = ReadIrradianceBank(vxPos + 0.5 * normal, 1).rgb;
    float lColor = length(color);
    if (lColor > 0.01) color *= log(lColor + 1) / lColor;
    return color;
}
vec3 readVolumetricBlocklight(vec3 vxPos) {
    if (!isInRange(vxPos)) return vec3(0);
    vec4 color = ReadIrradianceBank(vxPos, 1);
    return color.rgb / max(color.a, 0.0001);
}
#endif
