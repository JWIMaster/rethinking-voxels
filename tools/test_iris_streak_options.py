"""Exercise installed Iris's real scanner and property preprocessor (Java 25).
Bundled jcpp and Java class output go only under tools/.test-cache.
Run: python tools/test_iris_streak_options.py
"""
from pathlib import Path
import os
import subprocess
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PRISM = Path(os.environ['APPDATA']) / 'PrismLauncher'
IRIS = PRISM / 'instances/Skyblock Vanilla/minecraft/mods/iris-fabric-1.11.7+mc26.3.jar'
JAVA = PRISM / 'java/java-runtime-epsilon/bin'
CACHE = ROOT / 'tools/.test-cache/iris-options'
CACHE.mkdir(parents=True, exist_ok=True)
GUAVA = PRISM / 'libraries/com/google/guava/guava/33.6.0-jre/guava-33.6.0-jre.jar'
FASTUTIL = PRISM / 'libraries/it/unimi/dsi/fastutil/8.5.18/fastutil-8.5.18.jar'
SLF4J = PRISM / 'libraries/org/slf4j/slf4j-api/2.0.17/slf4j-api-2.0.17.jar'
for path in (IRIS, GUAVA, FASTUTIL, SLF4J, JAVA / 'java.exe', JAVA / 'javac.exe'):
    assert path.is_file(), path
with ZipFile(IRIS) as z:
    entry = next(n for n in z.namelist() if '/jcpp-' in n and n.endswith('.jar'))
    JCPP = CACHE / Path(entry).name
    JCPP.write_bytes(z.read(entry))
CLASSPATH = os.pathsep.join(map(str, (CACHE, IRIS, GUAVA, FASTUTIL, JCPP, SLF4J)))
subprocess.run([str(JAVA / 'javac.exe'), '-cp', CLASSPATH, '-d', str(CACHE),
                str(ROOT / 'tools/TestIrisStreakOptions.java')], check=True)
subprocess.run([str(JAVA / 'java.exe'), '-cp', CLASSPATH, 'TestIrisStreakOptions',
                str(ROOT / 'shaders'), str(CACHE / 'original-fixture')], check=True)
print('PASS installed Iris 1.11.7+mc26.3 option-discovery/property-allocation integration regression')
