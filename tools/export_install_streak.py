"""Export and install over the exact existing Skyblock Vanilla barrel shaderpack."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
PROFILE = Path(r"C:\Users\jwalr\AppData\Roaming\PrismLauncher\instances\Skyblock Vanilla")
NAME = "rethinking-voxels_r0.1-beta9_barrel-distortion.zip"
TARGET = PROFILE / "minecraft" / "shaderpacks" / NAME
SETTINGS = TARGET.with_name(NAME + ".txt")
IRIS = PROFILE / "minecraft" / "config" / "iris.properties"
EXPORT = ROOT / "rethinking-voxels_r0.1-beta9_barrel-distortion_kino-streak.zip"

assert PROFILE.is_dir() and TARGET.is_file() and SETTINGS.is_file() and IRIS.is_file()
assert TARGET.resolve() == Path(r"C:\Users\jwalr\AppData\Roaming\PrismLauncher\instances\Skyblock Vanilla\minecraft\shaderpacks\rethinking-voxels_r0.1-beta9_barrel-distortion.zip")
assert "shaderPack=" in IRIS.read_text(encoding="utf-8")
# Minecraft may change selection while this task runs. Do not overwrite a
# newer user selection: replace only the explicitly requested barrel pack.
assert not EXPORT.exists(), "Export already exists: inspect before replacing it."
print(f"Verified exact replacement target: {TARGET.resolve()}")
settings_before = SETTINGS.read_bytes()
iris_before = IRIS.read_bytes()
old_hash = hashlib.sha256(TARGET.read_bytes()).hexdigest()

# Preserve all the installed pack's non-shader support/placeholder assets,
# then include the entire current shader tree (never Git or development tools).
with ZipFile(TARGET) as baseline, ZipFile(EXPORT, "w", ZIP_DEFLATED, compresslevel=9) as output:
    for entry in baseline.infolist():
        if not entry.filename.startswith("shaders/") and not entry.is_dir():
            output.writestr(entry.filename, baseline.read(entry.filename))
    for path in sorted((ROOT / "shaders").rglob("*")):
        if path.is_file():
            output.write(path, path.relative_to(ROOT).as_posix())
    output.write(ROOT / "KinoStreak-README.md", "KinoStreak-README.md")

with ZipFile(EXPORT) as archive:
    assert archive.testzip() is None, "Archive CRC validation failed"
    names = archive.namelist()
    for world in ("world0", "world-1", "world1"):
        stages = [n for n in names if n.startswith(f"shaders/{world}/composite3") and n.endswith(".csh")]
        assert len(stages) == 20, (world, len(stages))
    assert "shaders/licenses/KinoStreak.txt" in names
    assert "shaders/lib/misc/lensDistortion.glsl" in names
    assert archive.read("shaders/program/final.glsl") == (ROOT / "shaders/program/final.glsl").read_bytes()
    for path in (ROOT / "shaders").rglob("*"):
        if path.is_file():
            assert archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes(), path

backup_dir = PROFILE / "shaderpack-backups" / ("before-kino-streak-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
backup_dir.mkdir(parents=True, exist_ok=False)
for source in (TARGET, SETTINGS, IRIS):
    shutil.copy2(source, backup_dir / source.name)
assert hashlib.sha256((backup_dir / NAME).read_bytes()).hexdigest() == old_hash
# Target is an explicitly verified absolute path; preserve both config files.
shutil.copy2(EXPORT, TARGET)
new_hash = hashlib.sha256(EXPORT.read_bytes()).hexdigest()
assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == new_hash
assert SETTINGS.read_bytes() == settings_before
# Do not assert the live config is byte-identical: Minecraft can legitimately
# save it while copying the zip. This script never writes that config.
if IRIS.read_bytes() != iris_before:
    print("Iris config was updated concurrently by Minecraft; left untouched.")
with ZipFile(TARGET) as installed:
    assert installed.testzip() is None

report = {
    "export": str(EXPORT), "installed": str(TARGET), "backup": str(backup_dir),
    "sha256": new_hash, "previous_sha256": old_hash,
    "settings_preserved": True, "iris_selection_preserved": True,
    "size_bytes": EXPORT.stat().st_size,
}
report_path = ROOT / "kino-streak-installation.json"
assert not report_path.exists(), "Installation report already exists"
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
