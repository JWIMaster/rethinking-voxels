# Travel stability and Cinematic Quality

## Travel stability fixes

These are renderer correctness changes, active even when Cinematic Quality is OFF:

- Immutable snapshots replace in-place GPU updates of rolling irradiance, voxel light candidates, screen-space light candidates, diffuse lighting and specular lighting. Workgroups no longer overwrite another workgroup's history while it is being read.
- Irradiance and candidate positions roll together with the integer camera origin. Newly exposed volume slabs, shader reloads and large teleports invalidate old data.
- Exact per-cell light records replace the camera-relative hash table; unrelated emitters cannot merge/unmerge as the camera crosses a block. Shared candidate deduplication also compares exact cell identities.
- Finite, deterministic power-of-two LOD selection replaces undefined/rounding-sensitive logarithms. Triangle bounds crossing the camera retain their required fine levels. Fine SDF queries cannot clamp unsupported world positions to a fine-grid edge.
- Rays stop on occupied signed-distance cells instead of marching backwards. Zero/nonfinite directions, atlas slab boundaries, missed hits and out-of-volume reads are guarded.
- Only original geometry colors donate during one-cell palette dilation; copied air colors cannot propagate again. Entity-light clumping preserves geometry flags and uses deterministic ownership.
- Normal/depth reprojection uses one writer per destination, rejects invalid history and retains hand-space semantics. Per-pixel lighting confidence uses linear depth rather than a loose device-depth test, with a smooth cache fallback instead of abrupt quality switching.
- SDF updates are refreshed when the rolling voxel origin moves instead of reusing an alternating stale band.

Existing barrel distortion, Classic Blue/Current flare modes, sky exclusion and pre-World-Blur streak generation remain intact.

## Single-switch movie quality

**Shader Pack Settings → Cinematic Quality → ON** (OFF by default).

Cinematic Quality uses compile-time quality overrides. Your ordinary settings remain stored; turning it OFF restores them. Other quality controls may still display those stored values while the movie overrides are active. Camera effects, color controls, resource-pack mode and your own shadow draw distance are not changed.

The mode selects a 256 × 128 × 256 voxel volume (±128 blocks horizontally), eight nested detail levels, full-resolution per-pixel block lighting, 128 light traces, 100 ray steps, 13 GI samples, maximum GI/shadow/reflection/detail/cloud/POM quality, 16× anisotropic filtering and TAA. SDF propagation runs every frame while the camera moves. Shadow filtering gets denser movie-only sampling.

Keep your shadow draw distance at 128 or more for movie shots: the movie volume is 128 blocks in every horizontal direction, and geometry is only voxelized inside the shadow render distance. At the default 160 the whole volume stays populated, and the smaller number of blocks per shadow texel makes shadows crisper than the old 512-block movie distance. A preset below 128 (for example the Low preset's 64) would leave the outer voxels empty; raise the draw distance before enabling cinematic quality.

This is intentionally very slow. At approximately 2560 × 1351, its custom images alone require about 1 GiB; Minecraft geometry, resource-pack textures, screen buffers and shadow buffers cost additional VRAM. Leave it OFF for normal play. If enabling it exhausts VRAM or stalls your GPU, turn it OFF. It is not enabled automatically on installation.

## Capture guidance and limits

Reload the updated shaderpack first. Load the route's Minecraft chunks and let voxel/SDF/GI history settle before recording; repeat after teleports or quality changes. Large volumes and low capture frame rates can make warm-up slow.

This remains a real-time voxel renderer, not an offline triangle path tracer. The switch does not freeze simulation, render many subframes per output frame, change Minecraft's render distance, or make unloaded chunks available. Nested voxel volumes still have finite coverage and spatial detail limits; forcing the finest level over the entire world would sample nonexistent geometry. These fixes target verified shader corruption and abrupt lighting fallbacks, not a promise of zero engine/chunk-streaming pop-in.

Validation on the RTX 3060 Ti: all 1,616 native programs / 2,432 stages pass across three dimensions and eight ordinary/Cinematic profile combinations, using actual Iris-expanded sources. All 824 compute dispatch declarations pass the installed Iris parser. Original option defaults remain discoverable, movie overrides and allocation dimensions agree, and no linked shader exceeds 7 of the GPU's 8 image units. Focused GPU suites pass rolling/history/GI, palette and entity ownership, 9,760 LOD queries, 168 atlas cases, traversal and deduplication, and 257 confidence cases plus 28 actual lighting renders. The existing flare suite also passes (1,640 compute compilations, 120 composite pairs, numerical/sky/color checks and Iris allocation regressions).

These are GPU/engine-parser checks, not an in-game route comparison. The largest movie resources were size-audited, not fully allocated and dispatched in a live Minecraft scene. In-game comparison is still required to assess remaining scene-specific artifacts.
