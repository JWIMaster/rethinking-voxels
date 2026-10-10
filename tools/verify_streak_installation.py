"""Read-only archive/profile verification; write a local installation receipt."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PROFILE = Path(r"C:\Users\jwalr\AppData\Roaming\PrismLauncher\instances\Skyblock Vanilla")
NAME = "rethinking-voxels_r0.1-beta9_barrel-distortion.zip"
TARGET = PROFILE / "minecraft/shaderpacks" / NAME
EXPORT = ROOT / "rethinking-voxels_r0.1-beta9_barrel-distortion_kino-streak.zip"
BACKUP = PROFILE / "shaderpack-backups/before-kino-streak-20261011-010707"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(TARGET) == sha(EXPORT)
assert sha(BACKUP / NAME) == "545c16690ae803b283a906c26f4fd7b24c8fa5598d03a877af9eb1145fa837a6"
settings = TARGET.with_name(NAME + ".txt")
assert settings.read_bytes() == (BACKUP / settings.name).read_bytes()
with ZipFile(TARGET) as z:
    assert z.testzip() is None
    for path in (ROOT / "shaders").rglob("*"):
        if path.is_file():
            assert z.read(path.relative_to(ROOT).as_posix()) == path.read_bytes(), path
    assert z.read("KinoStreak-README.md") == (ROOT / "KinoStreak-README.md").read_bytes()
    entries = len(z.namelist())
current_config = (PROFILE / "minecraft/config/iris.properties").read_text(encoding="utf-8")
selection = next(line.split("=", 1)[1] for line in current_config.splitlines() if line.startswith("shaderPack="))
report = {
    "export": str(EXPORT), "installed": str(TARGET), "backup": str(BACKUP),
    "sha256": sha(EXPORT), "previous_sha256": sha(BACKUP / NAME),
    "settings_preserved": True, "iris_config_modified_by_agent": False,
    "current_shaderpack": selection, "archive_entries": entries,
    "size_bytes": EXPORT.stat().st_size,
}
receipt = ROOT / "kino-streak-installation.json"
assert not receipt.exists()
receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
