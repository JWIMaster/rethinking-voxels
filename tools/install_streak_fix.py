"""Rebuild the corrected export and replace both known Skyblock Vanilla entries."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
PROFILE = Path(r"C:\Users\jwalr\AppData\Roaming\PrismLauncher\instances\Skyblock Vanilla")
PACKS = PROFILE / "minecraft/shaderpacks"
EXPORT_NAME = "rethinking-voxels_r0.1-beta9_barrel-distortion_kino-streak.zip"
NAMES = ("rethinking-voxels_r0.1-beta9_barrel-distortion.zip", EXPORT_NAME)
EXPORT = ROOT / EXPORT_NAME
TEMP = ROOT / (EXPORT_NAME + ".fixed.tmp")
IRIS = PROFILE / "minecraft/config/iris.properties"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
DOCS = ("KinoStreak-README.md", "Voxel-Stability-Cinematic-README.md")
assert ROOT == Path(r"C:\Users\jwalr\rethinking-voxels")
assert EXPORT.is_file() and PROFILE.is_dir() and IRIS.is_file()
assert not TEMP.exists()
for name in NAMES:
    target = PACKS / name
    assert target.is_file() and target.with_name(name + ".txt").is_file()
    assert target.resolve().parent == Path(r"C:\Users\jwalr\AppData\Roaming\PrismLauncher\instances\Skyblock Vanilla\minecraft\shaderpacks")
    print("Verified replacement target:", target.resolve())

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
BACKUP = PROFILE / "shaderpack-backups" / ("before-voxel-stability-cinematic-" + stamp)
BACKUP.mkdir(parents=True, exist_ok=False)
settings = {}
for name in NAMES:
    target = PACKS / name
    sidecar = target.with_name(name + ".txt")
    settings[name] = sidecar.read_bytes()
    shutil.copy2(target, BACKUP / name)
    shutil.copy2(sidecar, BACKUP / sidecar.name)
    assert sha(target) == sha(BACKUP / name)
shutil.copy2(IRIS, BACKUP / IRIS.name)
shutil.copy2(EXPORT, BACKUP / ("export-" + EXPORT_NAME))

with ZipFile(EXPORT) as old, ZipFile(TEMP, "w", ZIP_DEFLATED, compresslevel=9) as fixed:
    for info in old.infolist():
        if not info.is_dir() and not info.filename.startswith("shaders/") and info.filename not in DOCS:
            fixed.writestr(info.filename, old.read(info.filename))
    for path in sorted((ROOT / "shaders").rglob("*")):
        if path.is_file():
            fixed.write(path, path.relative_to(ROOT).as_posix())
    for name in DOCS:
        fixed.write(ROOT / name, name)

with ZipFile(TEMP) as fixed, ZipFile(EXPORT) as old:
    assert fixed.testzip() is None
    assert len(fixed.namelist()) == len(set(fixed.namelist()))
    for world in ("world0", "world-1", "world1"):
        stages = [n for n in fixed.namelist() if n.startswith(f"shaders/{world}/composite3") and n.endswith(".csh")]
        assert len(stages) == 20, (world, len(stages))
        assert not any(n.startswith(f"shaders/{world}/composite5") and n.endswith(".csh") for n in fixed.namelist()), "Old post-blur chain must not remain"
        for base, count in (("shadowcomp1", 6), ("prepare4", 3)):
            chain = [n for n in fixed.namelist() if n.startswith(f"shaders/{world}/{base}") and n.endswith(".csh")]
            assert len(chain) == count, (world, base, len(chain))
    assert b"#define CINEMATIC_MODE 0 //[0 1]" in fixed.read("shaders/lib/common.glsl"), "Cinematic must default OFF"
    for name in DOCS:
        assert fixed.read(name) == (ROOT / name).read_bytes()
    # The only final-pass change is removing the obsolete hash-table clear.
    # Verify the entire rest of the display/barrel program remains byte-exact.
    old_final = old.read("shaders/program/final.glsl").decode("utf-8")
    legacy_clear = '''    int index = int(gl_FragCoord.x) + int(viewWidth + 0.5) * int(gl_FragCoord.y);
    int clearPerFragment = 1 + 4 * (1<<18) / int(viewWidth * viewHeight + 0.5);
    if (index < 4 * (1<<18) / clearPerFragment) {
        for (int j = 0; j < clearPerFragment; j++) {
            globalLightHashMap[index * clearPerFragment + j] = uint(0);
        }
    }

'''
    clear_note = '''    // Per-voxel light records are cleared by Iris before voxelization, not
    // by a screen-size-dependent hash-table clear in this final display pass.

'''
    expected_final = old_final.replace(legacy_clear, clear_note)
    assert fixed.read("shaders/program/final.glsl").decode("utf-8") == expected_final, "Barrel/display code changed unexpectedly"
    assert fixed.read("shaders/lib/misc/lensDistortion.glsl") == old.read("shaders/lib/misc/lensDistortion.glsl")
    for path in (ROOT / "shaders").rglob("*"):
        if path.is_file():
            assert fixed.read(path.relative_to(ROOT).as_posix()) == path.read_bytes(), path

# All replaced paths are explicit, absolute, checked above, and backed up.
os.replace(TEMP, EXPORT)
new_hash = sha(EXPORT)
for name in NAMES:
    target = PACKS / name
    staging = target.with_name(name + ".fixed.tmp")
    assert not staging.exists()
    shutil.copy2(EXPORT, staging)
    assert sha(staging) == new_hash
    try:
        os.replace(staging, target)
    except PermissionError:
        # Minecraft's live ZipFS can allow writes but deny rename/delete of the
        # active zip on Windows. The existing archive was already backed up.
        shutil.copy2(staging, target)
        assert staging.resolve().parent == PACKS.resolve()
        assert staging.name in {n + ".fixed.tmp" for n in NAMES}
        print("Removing verified installation staging file:", staging.resolve())
        staging.unlink()
    assert sha(target) == new_hash
    assert target.with_name(name + ".txt").read_bytes() == settings[name]
    with ZipFile(target) as installed:
        assert installed.testzip() is None

selection = next(line.split("=", 1)[1] for line in IRIS.read_text(encoding="utf-8").splitlines() if line.startswith("shaderPack="))
report = {
    "export": str(EXPORT), "installed": [str(PACKS / n) for n in NAMES],
    "backup": str(BACKUP), "sha256": new_hash, "size_bytes": EXPORT.stat().st_size,
    "settings_preserved": True, "iris_config_modified": False, "current_shaderpack": selection,
    "fix": "Correct voxel/cache travel races, hash collisions, LOD/trace safety and abrupt lighting confidence fallback; add default-OFF single-switch Cinematic Quality while preserving camera effects, barrel distortion, flare corrections and all user settings.",
}
receipt = ROOT / ("kino-streak-installation-" + stamp + ".json")
assert not receipt.exists()
receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
