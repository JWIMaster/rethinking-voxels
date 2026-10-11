"""Restore the shipped shaderpack to the last anamorphic-streak (flare colour mode) release.

Deploys the verified color-mode archive captured immediately before the travel
stability / Cinematic Quality install, over both known Skyblock Vanilla entries.
User settings (.zip.txt sidecars) and the Iris shader selection are never touched.
"""
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime
import hashlib
import json
import os
import shutil

ROOT = Path(r"C:\Users\jwalr\rethinking-voxels")
PROFILE = Path(r"C:\Users\jwalr\AppData\Roaming\PrismLauncher\instances\Skyblock Vanilla")
PACKS = PROFILE / "minecraft/shaderpacks"
BACKUPS = PROFILE / "shaderpack-backups"
SOURCE_BACKUP = BACKUPS / "before-voxel-stability-cinematic-20261011-033803"
EXPORT_NAME = "rethinking-voxels_r0.1-beta9_barrel-distortion_kino-streak.zip"
NAMES = ("rethinking-voxels_r0.1-beta9_barrel-distortion.zip", EXPORT_NAME)
EXPORT = ROOT / EXPORT_NAME
IRIS = PROFILE / "minecraft/config/iris.properties"
ROLLBACK_SOURCE = SOURCE_BACKUP / ("export-" + EXPORT_NAME)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

assert ROOT == Path(r"C:\Users\jwalr\rethinking-voxels")
assert EXPORT.parent == ROOT and EXPORT.is_file() and IRIS.is_file()
assert ROLLBACK_SOURCE.is_file(), ROLLBACK_SOURCE
assert SOURCE_BACKUP.resolve().parent == BACKUPS.resolve()
for name in NAMES:
    target = PACKS / name
    assert target.is_file() and target.with_name(name + ".txt").is_file()
    assert target.resolve().parent == Path(r"C:\Users\jwalr\AppData\Roaming\PrismLauncher\instances\Skyblock Vanilla\minecraft\shaderpacks")
    print("Verified rollback target:", target.resolve())

# The source must be the flare colour-mode release, not the stability one.
with ZipFile(ROLLBACK_SOURCE) as z:
    assert z.testzip() is None
    names = z.namelist()
    assert "shaders/lib/vx/lightData.glsl" not in names
    assert "shaders/lib/misc/cinematicQuality.glsl" not in names
    common = z.read("shaders/lib/common.glsl").decode("utf-8")
    assert "#define STREAK_COLOR_MODE 0 //[0 1]" in common and "CINEMATIC_MODE" not in common
    assert "clearPerFragment" in z.read("shaders/program/final.glsl").decode("utf-8")
    for world in ("world0", "world-1", "world1"):
        assert len([n for n in names if n.startswith(f"shaders/{world}/composite3") and n.endswith(".csh")]) == 20
    print("Verified rollback source is the anamorphic-streak release:", sha(ROLLBACK_SOURCE))

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
SNAPSHOT = BACKUPS / ("rollback-of-voxel-stability-cinematic-" + stamp)
SNAPSHOT.mkdir(parents=True, exist_ok=False)
settings = {}
for name in NAMES:
    target = PACKS / name
    sidecar = target.with_name(name + ".txt")
    settings[name] = sidecar.read_bytes()
    shutil.copy2(target, SNAPSHOT / name)
    shutil.copy2(sidecar, SNAPSHOT / sidecar.name)
shutil.copy2(IRIS, SNAPSHOT / IRIS.name)
shutil.copy2(EXPORT, SNAPSHOT / ("export-" + EXPORT_NAME))
print("Preserved pre-rollback state in:", SNAPSHOT.resolve())

restored_hash = sha(ROLLBACK_SOURCE)
for name in NAMES:
    target = PACKS / name
    staging = target.with_name(name + ".rollback.tmp")
    assert not staging.exists()
    shutil.copy2(ROLLBACK_SOURCE, staging)
    assert sha(staging) == restored_hash
    try:
        os.replace(staging, target)
    except PermissionError:
        # Minecraft's live ZipFS can deny rename/delete of the active zip on
        # Windows while still allowing an in-place overwrite.
        shutil.copy2(staging, target)
        assert staging.resolve().parent == PACKS.resolve()
        assert staging.name in {n + ".rollback.tmp" for n in NAMES}
        print("Removing verified rollback staging file:", staging.resolve())
        staging.unlink()
    assert sha(target) == restored_hash
    assert target.with_name(name + ".txt").read_bytes() == settings[name]
    with ZipFile(target) as installed:
        assert installed.testzip() is None

shutil.copy2(ROLLBACK_SOURCE, EXPORT)
assert sha(EXPORT) == restored_hash

selection = next(line.split("=", 1)[1] for line in IRIS.read_text(encoding="utf-8").splitlines() if line.startswith("shaderPack="))
report = {
    "export": str(EXPORT), "installed": [str(PACKS / n) for n in NAMES],
    "rollback_source": str(ROLLBACK_SOURCE), "pre_rollback_snapshot": str(SNAPSHOT),
    "sha256": restored_hash, "size_bytes": ROLLBACK_SOURCE.stat().st_size,
    "settings_preserved": True, "iris_config_modified": False, "current_shaderpack": selection,
    "restored": "Last anamorphic-streak release: flare colour mode (Current/Custom tint or Classic Blue) with sky rejection, pre-World-Blur streak generation and chromatic-aberration alignment.",
    "removed_from_install": "Travel stability fixes and the Cinematic Quality switch; preserved in git branch wip-voxel-stability-cinematic and in the pre-rollback snapshot.",
}
receipt = ROOT / ("rollback-to-streak-release-" + stamp + ".json")
assert not receipt.exists()
receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
