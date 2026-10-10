# KinoStreak for Rethinking Voxels

GLSL/Iris adaptation of [KinoStreak](https://github.com/keijiro/KinoStreak) by Keijiro Takahashi (MIT license retained under `shaders/licenses/KinoStreak.txt`). The existing barrel-distortion effect is unchanged.

## Controls

**Shader Pack Settings → Camera → KinoStreak / Anamorphic Streaks**

- **Anamorphic Streaks:** enabled by default, independent of ordinary Bloom.
- **Flare Color Mode:** **Current / Custom Tint** (default) preserves the existing appearance and your RGB tint sliders. **Classic Blue** converts highlight strength to a fixed cool-blue flare (`0.25, 0.55, 1.0`) regardless of source-light color; custom RGB tint sliders are ignored in this mode. Switching modes does not change threshold, stretch, intensity, sky rejection or pre-blur generation.
- **Brightness Threshold:** 1.00 linear HDR by default. Increase to limit flares to brighter highlights.
- **Stretch:** 0.75 by default. Larger values mix in wider horizontal levels.
- **Intensity:** 30% (0.30) by default. Zero disables filtering/image allocations.
- **Custom Tint:** red 0.55, green 0.55, blue 1.00 (KinoStreak's original defaults); applies only to Current / Custom Tint mode.

Disable Anamorphic Streaks to remove atlas allocations and filtering. The empty compute entry points then return without processing pixels.

## Integration

The compute chain runs before composite3's World Blur raster pass, so the HDR streak source is not displaced by World Blur's chromatic-aberration sampling. The finished atlas is still added at composite5 before tone mapping. Filtering consists of: thresholded two-tap half-height prefilter, six-tap horizontal downsampling, and lerp-based reconstruction. The prefilter is fused into the first downsample, with interpolation of thresholded prefilter texels to retain the original filter order. Horizontal mip levels are packed in two half-height RGBA16F images; bounds are clamped per level to prevent atlas bleed or screen-edge wrapping. Vertical resolution stays unchanged after the initial prefilter. At 1920×1080 the two images consume about 16 MiB.

The chain stops at an 8–16 pixel-wide level (maximum ten levels). Unused fixed entry points do no image work. Explicit `allowConcurrentCompute=false` preserves Iris's memory barriers between dependent dispatches. Streaks are composed additively with the selected flare color and intensity×5 factor before color grading, tone mapping, and TAA. The existing final barrel distortion therefore bends the streaks together with the scene. Only depth-bearing world pixels seed the streak: pure sky, sun/moon and volumetric clouds are excluded before source interpolation. Distant Horizons depth allows distant terrain to contribute even when vanilla depth is empty. Sky reflected on depth-bearing geometry can still contribute, and streaks from world lights may naturally extend across the sky. Motion blur (when enabled) is already present in the source. Bloom-fog amplification applied inside composite3 no longer feeds streak generation while World Blur is enabled.

## Sky and chromatic World Blur correction

Sky rejection is per original source texel, before vertical filtering, brightness thresholding or horizontal interpolation. This prevents a bright sky texel from being blended into a neighboring terrain pixel and falsely seeding a streak at the skyline. The chain moved from composite5 to composite3; all old composite5 compute entry points were removed. World Blur, its chromatic/anamorphic options, and final camera chromatic aberration remain available and unchanged, but their displaced RGB samples are no longer reused as streak sources.

## Overlay artifact fix

The first export incorrectly used only `#if defined KINO_STREAK` references for the boolean option. Iris recognizes boolean shader options only when referenced by a plain `#ifdef` or `#ifndef`. Consequently GLSL enabled the effect but the properties preprocessor omitted both atlas allocations; the unset streak sampler used texture unit zero and stretched the left half of the scene over the image. The fix explicitly activates the boolean option for Iris discovery. Compute and composition now also reject resources whose dimensions do not match the expected half-height atlases, preventing that fallback from drawing an overlay or writing unrelated images. This is a texture-allocation/binding fix, not a reduction in depth-of-field or motion-blur settings.

## Export and installation

The exported zip has `shaders/` at its root. In Skyblock Vanilla the corrected version replaces both `rethinking-voxels_r0.1-beta9_barrel-distortion.zip` and the currently used `rethinking-voxels_r0.1-beta9_barrel-distortion_kino-streak.zip`, preserving each `.zip.txt` options file and leaving the current Iris shader selection unchanged. The previous installed zip and settings are backed up outside the shaderpacks directory. If Minecraft is already open, reload shaders (Iris's Reload Shaders key, normally R) or toggle shaders off/on to load the replaced archive.

## Validation

Validated on NVIDIA GeForce RTX 3060 Ti (OpenGL 4.3): 1,640 compute programs and 120 original composite3/composite5 vertex/fragment pairs compiled, linked and exercised, including 20 option/profile variants and the current user's chromatic/anamorphic depth-of-field settings. GPU results were checked against an independent CPU filter reference, including black/subthreshold inputs, HDR constants, random HDR values, horizontal impulses, edge clamping, odd/tiny dimensions, all ten pyramid levels, and effect-disabled paths. Additional depth tests prove that all-sky HDR input and bright sky next to subthreshold terrain generate exactly zero streak RGB, while world emitters and Distant Horizons terrain remain valid sources. A GPU two-pass test renders the pack's actual chromatic World Blur function: feeding that displaced output into the old streak placement reproduces RGB-split streaks, while the new pre-blur source remains aligned and achromatic. Streak atlases are bit-identical across WORLD_BLUR=0/1/2 and chromatic on/off when given the same raw source. Forty-eight composition checks cover both color modes, colored/negative/mixed inputs, custom tint independence in Classic Blue, unchanged Current-mode output, and missing-sampler rejection. The installed option scanner confirms default mode 0, selectable mode 1, and both allowed values. The actual installed Iris 1.11.7 option scanner and properties preprocessor also reproduce the original missing-allocation bug and verify corrected default/ON/OFF/zero-intensity allocation behavior. GPU regression cases verify that an unset sampler bound to the full-height scene contributes exactly zero, and wrong-sized compute images are not written. These are standalone engine-parser/GPU checks, not an in-game visual confirmation; reload shaders after replacing an archive while Minecraft is open.

## Development checks

- `python tools/generate_streak_passes.py` verifies/generates the three dimensions' entry points.
- `python tools/test_kino_streak.py` performs GPU compilation and numerical filter tests; requires `moderngl`, `glcontext`, and `numpy`.
- `python tools/test_iris_streak_options.py` uses the installed Iris scanner/preprocessor and bundled JCPP to verify real option discovery and custom-image declarations; requires this machine's Java 25/Prism dependencies.
- `python tools/install_streak_fix.py` rebuilds the corrected zip and replaces both affected Skyblock Vanilla entries, with backups and checksum validation.
