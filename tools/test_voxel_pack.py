"""Compile/link every pack entry point on the real NVIDIA OpenGL driver.

Run: python tools/test_voxel_pack.py [--profiles LOW HIGH ULTRA current]
     [--cinematic 0 1] [--affected-only] [--iris-options]

No scene/voxel textures are allocated and NO programs are dispatched. This
checks real expanded shader sources, not translated algorithms. Original
versions (including GLSL 430 compatibility) are kept. Failures are retained
under tools/.test-cache/voxel-pack with exact expanded stage sources.
"""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import moderngl
import test_kino_streak as streak

ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / 'shaders'
CACHE = ROOT / 'tools/.test-cache/voxel-pack'
WORLDS = ('world0', 'world-1', 'world1')
PROFILE = Path(os.environ.get('APPDATA', '')) / 'PrismLauncher/instances/Skyblock Vanilla/minecraft'
SIDECAR = PROFILE / 'shaderpacks/rethinking-voxels_r0.1-beta9_barrel-distortion_kino-streak.zip.txt'
AFFECTED = ('shadow', 'shadowcomp', 'prepare', 'final', 'gbuffers', 'deferred')
RENDER_STAGES = ('NONE SKY SUNSET CUSTOM_SKY SUN MOON STARS VOID TERRAIN_SOLID '
                 'TERRAIN_CUTOUT_MIPPED TERRAIN_CUTOUT ENTITIES BLOCK_ENTITIES DESTROY '
                 'OUTLINE DEBUG HAND_SOLID TERRAIN_TRANSLUCENT TRIPWIRE PARTICLES CLOUDS '
                 'RAIN_SNOW WORLD_BORDER HAND_TRANSLUCENT').split()


def current_options():
    values = {}
    for line in SIDECAR.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, val = line.split('=', 1)
        values[name] = True if val == 'true' else False if val == 'false' else val
    # Request these sensitive paths even if omitted because they equal defaults.
    values.update(GI_STRENGTH='2', RP_MODE='3', WORLD_BLUR='2', STREAK_COLOR_MODE='1', SHADOW_QUALITY='5')
    return values


def environment(ctx):
    out = ['#define MC_OS_WINDOWS', '#define MC_GL_VENDOR_NVIDIA', '#define MC_GL_RENDERER_GEFORCE',
           '#define IRIS_REQUIRES_SEPARATE_ENTITY_DRAWS', '#define IRIS_HAS_TRANSLUCENCY_SORTING',
           '#define IRIS_HAS_CONNECTED_TEXTURES', '#define IRIS_TAG_SUPPORT 2', '#define MAX_COLOR_BUFFERS 16',
           '#define MC_NORMAL_MAP', '#define MC_SPECULAR_MAP', '#define MC_RENDER_QUALITY 1.0',
           '#define MC_SHADOW_QUALITY 1.0', '#define MC_HAND_DEPTH 0.125', '#define MC_MIPMAP_LEVEL 4',
           '#define MC_GL_VERSION 430', '#define MC_GLSL_VERSION 430',
           '#define IRIS_FEATURE_CUSTOM_IMAGES', '#define IRIS_FEATURE_SSBO',
           '#define IRIS_FEATURE_COMPUTE_SHADERS', '#define IRIS_FEATURE_BLOCK_EMISSION_ATTRIBUTE',
           '#define IRIS_FEATURE_HIGHER_SHADOWCOLOR']
    out += [f'#define MC_RENDER_STAGE_{name} {i}' for i, name in enumerate(RENDER_STAGES)]
    # Confirmed against installed Iris StandardMacros bytecode.
    out += ['#define DH_BLOCK_UNKNOWN 0', '#define DH_BLOCK_LEAVES 1',
            '#define DH_BLOCK_LAVA 6', '#define DH_BLOCK_WATER 12',
            '#define DH_BLOCK_GRASS 13', '#define DH_BLOCK_ILLUMINATED 15']
    out += ['#define MC_' + ext for ext in sorted(ctx.extensions)]
    return '\n'.join(out) + '\n'


def shader_source(path, options, env, actual_stage_root=None):
    if path.name.startswith('dh_'):
        # DH entry points are only loaded when the engine's DH module is active.
        options = dict(options, DISTANT_HORIZONS=True)
        env += '#define DISTANT_HORIZONS\n#define DISTANT_HORIZONS_TEXTURES\n'
    if actual_stage_root is not None:
        # Actual Iris rewrites ONLY annotated option-definition locations.
        # This preserves conditional cinematic derived #undef/#define overrides.
        text = (actual_stage_root / path.relative_to(SHADERS)).read_text(encoding='utf-8')
        env += '#define IS_IRIS\n#define IRIS_VERSION 11107\n#define MC_VERSION 260300\n'
    else:
        text = streak.source(path, options)
    at = text.index('\n') + 1
    interfaces = ''
    # Iris's line/DH transformers supply these engine interface declarations;
    # they are not replacements for any shader algorithm or native builtins.
    if path.name == 'gbuffers_line.vsh':
        interfaces += 'uniform mat4 projectionMatrix; uniform mat4 modelViewMatrix;\nin vec3 vaPosition; in vec3 vaNormal;\n'
    if path.name in ('dh_terrain.vsh', 'dh_water.vsh'):
        interfaces += 'in int dhMaterialId;\n'
    return text[:at] + env + interfaces + text[at:]


def capability_limits(ctx):
    # Core info sometimes omits compute/image enums; query current real context.
    opengl = ctypes.WinDLL('opengl32')
    get = opengl.glGetIntegerv
    get.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
    values = {
        'GL_MAX_IMAGE_UNITS': 0x8F38, 'GL_MAX_COMBINED_IMAGE_UNIFORMS': 0x90CF,
        'GL_MAX_COMPUTE_IMAGE_UNIFORMS': 0x91BD, 'GL_MAX_FRAGMENT_IMAGE_UNIFORMS': 0x90CE,
        'GL_MAX_VERTEX_IMAGE_UNIFORMS': 0x90CA, 'GL_MAX_GEOMETRY_IMAGE_UNIFORMS': 0x90CD,
        'GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS': 0x90EB, 'GL_MAX_COMPUTE_SHARED_MEMORY_SIZE': 0x8262,
        'GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS': 0x90DD, 'GL_MAX_3D_TEXTURE_SIZE': 0x8073,
        'GL_MAX_TEXTURE_SIZE': 0x0D33, 'GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS': 0x8B4D,
        'GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS': 0x8B4C, 'GL_MAX_TEXTURE_IMAGE_UNITS': 0x8872,
        'GL_CONTEXT_PROFILE_MASK': 0x9126,
    }
    result = {}
    for name, enum in values.items():
        dest = ctypes.c_int()
        get(enum, ctypes.byref(dest))
        result[name] = dest.value
    return result


def stage_sets(world, affected_only=False):
    groups = {}
    for path in sorted((SHADERS / world).iterdir()):
        if path.suffix not in ('.vsh', '.gsh', '.fsh', '.csh'):
            continue
        if affected_only and not path.stem.startswith(AFFECTED):
            continue
        if path.suffix == '.csh':
            groups['compute:' + path.stem] = {'compute_shader': path}
        else:
            groups.setdefault('raster:' + path.stem, {})[
                {'.vsh': 'vertex_shader', '.gsh': 'geometry_shader', '.fsh': 'fragment_shader'}[path.suffix]] = path
    for name, group in groups.items():
        if name.startswith('raster:'):
            assert 'vertex_shader' in group and 'fragment_shader' in group, (world, name, group)
    return groups


def check_wrapper_chains():
    for world in WORLDS:
        groups = stage_sets(world)
        chains = {}
        for name in groups:
            if not name.startswith('compute:'):
                continue
            stem = name.split(':', 1)[1]
            m = re.fullmatch(r'(.+)_([a-z])', stem)
            if m:
                chains.setdefault(m.group(1), set()).add(m.group(2))
        for base, suffixes in chains.items():
            assert 'compute:' + base in groups, ('missing base', world, base)
            expected = {chr(c) for c in range(ord('a'), ord(max(suffixes)) + 1)}
            assert suffixes == expected, ('compute suffix gap', world, base, suffixes)
    return True


def source_snapshot():
    digest = hashlib.sha256()
    for path in sorted(SHADERS.rglob('*')):
        if path.is_file() and path.suffix in ('.glsl', '.fsh', '.vsh', '.gsh', '.csh', '.properties'):
            digest.update(path.relative_to(SHADERS).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


PROPERTIES_JAVA = r'''
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.lang.reflect.Constructor;
import java.nio.file.*;
import java.util.*;
import net.irisshaders.iris.helpers.StringPair;
import net.irisshaders.iris.shaderpack.include.*;
import net.irisshaders.iris.shaderpack.option.ShaderPackOptions;
import net.irisshaders.iris.shaderpack.preprocessor.PropertiesPreprocessor;
import net.irisshaders.iris.shaderpack.preprocessor.JcppProcessor;
public class VoxelPackProperties {
 public static void main(String[] args) throws Exception {
  Path root=Path.of(args[0]);
  var nodes=new HashMap<AbsolutePackPath,FileNode>();
  var todo=new ArrayDeque<AbsolutePackPath>();
  try(var files=Files.walk(root)) {
   files.filter(Files::isRegularFile).filter(p->p.toString().matches(".*\\.(vsh|fsh|gsh|csh)$"))
    .forEach(p->todo.add(AbsolutePackPath.fromAbsolutePath("/"+root.relativize(p).toString().replace('\\','/'))));
  }
  while(!todo.isEmpty()) {
   var path=todo.removeFirst(); if(nodes.containsKey(path)) continue;
   var node=new FileNode(path,ImmutableList.copyOf(Files.readString(path.resolved(root)).split("\\R")));
   nodes.put(path,node);todo.addAll(node.getIncludes().values());
  }
  Constructor<IncludeGraph> ctor=IncludeGraph.class.getDeclaredConstructor(ImmutableMap.class,ImmutableMap.class);
  ctor.setAccessible(true);
  // Bypass only Minecraft-dependent debug-IO construction. Actual Iris include
  // nodes, option discovery and property/JCPP preprocessing remain untouched.
  var graph=ctor.newInstance(ImmutableMap.copyOf(nodes),ImmutableMap.of());
  var props=new Properties();try(var reader=Files.newBufferedReader(Path.of(args[1]))) {props.load(reader);}
  var changes=new HashMap<String,String>();props.forEach((k,v)->changes.put(k.toString(),v.toString()));
  var options=new ShaderPackOptions(graph,changes);
  // Derived cinematic aliases must never hide/change original annotated defaults.
  var pattern=java.util.regex.Pattern.compile("(?m)^\\s*#define\\s+(\\w+)\\s+([^\\s/]+)\\s*//\\s*\\[[^\\]]+\\]");
  var matcher=pattern.matcher(Files.readString(root.resolve("lib/common.glsl")));int checked=0;
  while(matcher.find()) {
   String name=matcher.group(1),value=matcher.group(2);
   var original=options.getOptionSet().getStringOptions().get(name);
   if(original==null || !original.getOption().getDefaultValue().equals(value))
    throw new AssertionError("Original option default not discoverable/preserved: "+name+"="+value);
   checked++;
  }
  if(!options.getOptionSet().getBooleanOptions().containsKey("KINO_STREAK")) throw new AssertionError("KINO_STREAK toggle hidden");
  var drawOption=options.getOptionSet().getStringOptions().get("shadowDistance");
  if(drawOption==null || !drawOption.getOption().getDefaultValue().equals("128.0")) throw new AssertionError("Original shadowDistance option/default128.0 hidden/changed");
  var movie=options.getOptionSet().getStringOptions().get("CINEMATIC_MODE");
  if(movie==null || !movie.getOption().getDefaultValue().equals("0") || !movie.getOption().getAllowedValues().equals(List.of("0","1"))) throw new AssertionError("CINEMATIC_MODE must staydefault0 withallowed0/1");
  if(!options.getOptionValues().getStringValueOrDefault("CINEMATIC_MODE").equals(changes.get("CINEMATIC_MODE"))) throw new AssertionError("Cinematic override notappliedbyIris");
  var rawProperties=new Properties();rawProperties.load(new java.io.StringReader(Files.readString(root.resolve("shaders.properties"))));
  if(!List.of(rawProperties.getProperty("screen","").split("\\s+")).contains("CINEMATIC_MODE")) throw new AssertionError("Cinematic switch notinrootmenu");
  var boolPattern=java.util.regex.Pattern.compile("(?m)^[\\t ]*(//[\\t ]*)?#define[\\t ]+(PER_PIXEL_LIGHT|LONGER_LIGHT_LISTS|TRANSLUCENT_LIGHT_TINT|ENTITY_VOXELIZATION|PLAYER_VOXELIZATION|FOLIAGE_VOXELIZATION|VOXEL_RT_REFLECTIONS|KINO_STREAK)[\\t ]*(?://[^\\r\\n]*)?$");
  var boolMatcher=boolPattern.matcher(Files.readString(root.resolve("lib/common.glsl")));int checkedBools=0;var boolFailures=new ArrayList<String>();
  while(boolMatcher.find()) {
   var bool=options.getOptionSet().getBooleanOptions().get(boolMatcher.group(2));
   if(bool==null || bool.getOption().getDefaultValue()!=(boolMatcher.group(1)==null)) boolFailures.add(boolMatcher.group(2)+" expected="+(boolMatcher.group(1)==null)+" actual="+(bool==null?"MISSING":bool.getOption().getDefaultValue()));
   checkedBools++;
  }
  if(checkedBools!=8) throw new AssertionError("Expected8originalpublicboolcontrols, got"+checkedBools);
  if(!boolFailures.isEmpty()) throw new AssertionError("Original publicboolcontrols hidden/changed: "+boolFailures);
  System.out.println("PASS actual Iris original option defaults: "+checked+" annotated numeric controls, KINO_STREAK/CINEMATIC_MODE alive");
  var env=new ArrayList<StringPair>();
  env.add(new StringPair("IS_IRIS",""));env.add(new StringPair("IRIS_VERSION","11107"));
  env.add(new StringPair("MC_VERSION","260300"));
  env.add(new StringPair("IRIS_FEATURE_BLOCK_EMISSION_ATTRIBUTE",""));
  env.add(new StringPair("IRIS_FEATURE_HIGHER_SHADOWCOLOR",""));
  String output=PropertiesPreprocessor.preprocessSource(Files.readString(root.resolve("shaders.properties")),options,env);
  Files.writeString(Path.of(args[2]),output);
  var originalValues=new Properties();
  for(var entry:options.getOptionSet().getStringOptions().entrySet()) originalValues.setProperty(entry.getKey(),options.getOptionValues().getStringValueOrDefault(entry.getKey()));
  for(var entry:options.getOptionSet().getBooleanOptions().entrySet()) originalValues.setProperty(entry.getKey(),Boolean.toString(options.getOptionValues().getBooleanValueOrDefault(entry.getKey())));
  try(var writer=Files.newBufferedWriter(Path.of(args[2]+".original-values.properties"))) {originalValues.store(writer,"Original user/profile values beforecinematicruntimealiases");}
  Path metadata=Path.of(args[2]+".compute-metadata");Files.createDirectories(metadata);
  var included=new IncludeProcessor(options.getIncludes());
  Path stageRoot=Path.of(args[2]+".expanded-stages");Files.createDirectories(stageRoot);
  for(var path:nodes.keySet()) {
   if(!path.getPathString().matches(".*\\.(vsh|fsh|gsh|csh)$")) continue;
   String expanded=String.join("\n",included.getIncludedFile(path));
   Path stageFile=stageRoot.resolve(path.getPathString().substring(1));Files.createDirectories(stageFile.getParent());
   Files.writeString(stageFile,expanded);
   if(!path.getPathString().endsWith(".csh")) continue;
   StringBuilder probes=new StringBuilder("\n");
   for(String name:List.of("VX_VOL_SIZE","VOXEL_DETAIL_AMOUNT","SDF_UPDATE_INTERVAL","GI_STRENGTH","GI_SAMPLE_COUNT","BLOCKLIGHT_RESOLUTION","MAX_TRACE_COUNT","RT_STEPS","LIGHT_TRACE_LENGTH","SHADOW_QUALITY","DETAIL_QUALITY","CLOUD_QUALITY","LIGHTSHAFT_QUALI_DEFINE","WATER_REFLECT_QUALITY","BLOCK_REFLECT_QUALITY","SSAO_QUALI_DEFINE","ANISOTROPIC_FILTER","POM_QUALITY","POM_DISTANCE","TAA_MODE","ENTITY_SHADOWS_DEFINE","WORLD_BLUR","WB_DOF_I","WB_DOF_FOCUS","BARREL_DISTORTION","BARREL_DISTORTION_MODE","CHROMA_ABERRATION","STREAK_COLOR_MODE","STREAK_THRESHOLD","STREAK_STRETCH","STREAK_INTENSITY","LENSFLARE_MODE","LENSFLARE_I","T_EXPOSURE","T_SATURATION","MOTION_BLURRING_STRENGTH")) {
    probes.append("#ifdef ").append(name).append("\nconst float RV_TEST_EFFECTIVE_").append(name).append(" = ").append(name).append(";\n#else\nconst float RV_TEST_EFFECTIVE_").append(name).append(" = -9999;\n#endif\n");
   }
   probes.append("const float RV_TEST_EFFECTIVE_shadowDistance = shadowDistance;\n");
   for(String name:List.of("PER_PIXEL_LIGHT","PER_BLOCK_LIGHT","LONGER_LIGHT_LISTS","TRANSLUCENT_LIGHT_TINT","GI","BLOCKLIGHT_HIGHLIGHT","ENTITY_VOXELIZATION","PLAYER_VOXELIZATION","FOLIAGE_VOXELIZATION","VOXEL_RT_REFLECTIONS","KINO_STREAK","MOTION_BLURRING","WB_CHROMATIC","WB_ANAMORPHIC","WB_FOV_SCALED","BLOOM","COLORGRADING")) {
    probes.append("#ifdef ").append(name).append("\nconst int RV_TEST_BOOLEAN_").append(name).append(" = 1;\n#else\nconst int RV_TEST_BOOLEAN_").append(name).append(" = 0;\n#endif\n");
   }
   String processed=JcppProcessor.glslPreprocessSource(expanded+probes.toString(),env);
   StringBuilder kept=new StringBuilder();
   for(String line:processed.split("\\R")) {
    if(line.contains("workGroups") || line.contains("local_size_") || line.contains("const ivec3 voxelVolumeSize") || line.contains("const float shadowDistance") || line.contains("RV_TEST_") || line.contains("uniform")&&line.contains("image")) kept.append(line).append("\n");
   }
   Files.writeString(metadata.resolve(path.getPathString().replace('/','_')+".txt"),kept.toString());
  }
 }
}
'''


def prepare_property_preprocessor():
    prism = Path(os.environ['APPDATA']) / 'PrismLauncher'
    java = prism / 'java/java-runtime-epsilon/bin'
    iris = prism / 'instances/Skyblock Vanilla/minecraft/mods/iris-fabric-1.11.7+mc26.3.jar'
    jars = [iris, prism/'libraries/com/google/guava/guava/33.6.0-jre/guava-33.6.0-jre.jar',
            prism/'libraries/it/unimi/dsi/fastutil/8.5.18/fastutil-8.5.18.jar',
            prism/'libraries/org/slf4j/slf4j-api/2.0.17/slf4j-api-2.0.17.jar',
            ROOT/'tools/.test-cache/iris-options/jcpp-1.4.14.jar']
    for path in [java/'java.exe', java/'javac.exe', *jars]:
        assert path.is_file(), path
    source = CACHE/'VoxelPackProperties.java'
    if source.exists(): source.read_text(encoding='utf-8')
    source.write_text(PROPERTIES_JAVA,encoding='utf-8')
    classpath = os.pathsep.join(str(p) for p in [CACHE,*jars])
    subprocess.run([str(java/'javac.exe'),'-cp',classpath,'-d',str(CACHE),str(source)],check=True)
    return java/'java.exe', classpath


def audit_properties(java, classpath, variant, options, caps):
    import math
    settings = CACHE/(variant+'.options.properties')
    if settings.exists(): settings.read_text(encoding='utf-8')
    settings.write_text(''.join(f'{key}={str(val).lower() if isinstance(val,bool) else val}\n' for key,val in options.items()),encoding='utf-8')
    output = CACHE/(variant+'.preprocessed.properties')
    subprocess.run([str(java),'-cp',classpath,'VoxelPackProperties',str(SHADERS),str(settings),str(output)],check=True)
    props = {}
    for line in output.read_text(encoding='utf-8').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key,value=line.split('=',1);props[key.strip()]=value.strip()
    images=[];errors=[];warnings=[];total=0
    for key,value in props.items():
        if not key.startswith('image.'): continue
        tokens=value.split()
        if len(tokens) not in (8,9):
            errors.append(f'{key}: malformed declaration {value}');continue
        sampler,external,internal,ty,clear,relative,*shape=tokens
        if clear.lower() not in ('true','false') or relative.lower() not in ('true','false'):
            errors.append(f'{key}: malformed boolean metadata {value}');continue
        if relative.lower()=='true':
            if len(shape)!=2: errors.append(f'{key}: relative dimensions must be 2D');continue
            dims=[int(float(shape[0])*2560),int(float(shape[1])*1351)]
        else: dims=[int(float(x)) for x in shape]
        limit=caps['GL_MAX_3D_TEXTURE_SIZE'] if len(dims)==3 else caps['GL_MAX_TEXTURE_SIZE']
        if any(d<=0 or d>limit for d in dims): errors.append(f'{key}: dimensions {dims} exceed native limit {limit}')
        fmt=internal.upper()
        if fmt.endswith('UI') and (not external.upper().endswith('_INTEGER') or ty.upper() not in ('UNSIGNED_INT','UNSIGNED_BYTE','UNSIGNED_SHORT')):
            errors.append(f'{key}: unsigned integer image format/type mismatch {value}')
        elif fmt.endswith('I') and not fmt.endswith('UI') and (not external.upper().endswith('_INTEGER') or ty.upper() not in ('INT','BYTE','SHORT')):
            errors.append(f'{key}: signed integer image format/type mismatch {value}')
        elif fmt.endswith('F') and (external.upper().endswith('_INTEGER') or ty.upper() not in ('FLOAT','HALF_FLOAT')):
            errors.append(f'{key}: floating image format/type mismatch {value}')
        m=re.fullmatch(r'(R|RG|RGB|RGBA)(8|16|32)(?:F|I|UI|_SNORM)?',fmt)
        bpp={'R11F_G11F_B10F':4,'RGB10_A2':4}.get(fmt)
        if m: bpp=len(m.group(1))*int(m.group(2))//8
        if bpp is None: errors.append(f'{key}: unknown internal-format byte size {internal}');continue
        size=math.prod(dims)*bpp;total+=size
        images.append({'name':key,'sampler':sampler,'external':external,'internal':internal,'type':ty,'dimensions':dims,'bytes':size})
    # Global image pool size is NOT GL_MAX_IMAGE_UNITS. Iris assigns only each
    # linked program's active image uniforms; those are checked after native link.
    buffers=[]
    for key,value in props.items():
        if key.startswith('bufferObject.'):
            count=int(value.split()[0]);total+=count;buffers.append({'name':key,'bytes':count})
    metadata_folder=Path(str(output)+'.compute-metadata')
    metadata_files=list(metadata_folder.glob('*.txt'))
    expected_compute=len(list(SHADERS.glob('world*/*.csh')))
    if len(metadata_files)!=expected_compute: errors.append(f'Actual JCPP compute metadata count {len(metadata_files)} != {expected_compute}')
    directive_counts={'workGroups':0,'workGroupsRender':0}
    volume_sizes=set()
    effective_values={}; boolean_values={}; shadow_distances=set()
    original_values={}
    for line in Path(str(output)+'.original-values.properties').read_text(encoding='utf-8').splitlines():
        if '=' in line and not line.startswith('#'):
            key,value=line.split('=',1);original_values[key]=value
    literal=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    for file in metadata_files:
        text=file.read_text(encoding='utf-8')
        for name,value in re.findall(r'RV_TEST_EFFECTIVE_(\w+)\s*=\s*('+literal+r')\s*;',text):
            effective_values.setdefault(name,set()).add(float(value))
        for name,value in re.findall(r'RV_TEST_BOOLEAN_(\w+)\s*=\s*([01])\s*;',text):
            boolean_values.setdefault(name,set()).add(int(value))
        for value in re.findall(r'const\s+float\s+shadowDistance\s*=\s*('+literal+r')\s*;',text):
            shadow_distances.add(float(value))
        for match in re.findall(r'const\s+ivec3\s+voxelVolumeSize\s*=\s*ivec3\s*\(([^)]+)\)',text):
            values=tuple(int(x.strip()) for x in match.split(','))
            volume_sizes.add(values)
        for name,constructor,dimensions in [('workGroups','ivec3',3),('workGroupsRender','vec2',2)]:
            matches=re.findall(r'const\s+\w+\s+'+name+r'\s*=\s*([^;]+);',text)
            if len(matches)>1: errors.append(f'{file.name}: duplicate active {name} directives')
            for value in matches:
                m=re.fullmatch(constructor+r'\s*\((.*)\)',value.strip())
                if not m: errors.append(f'{file.name}: Iris cannot parse {name}={value}');continue
                arguments=[x.strip() for x in m.group(1).split(',')]
                if len(arguments)!=dimensions or not all(re.fullmatch(literal,x) for x in arguments):
                    errors.append(f'{file.name}: {name} requires {dimensions} numeric literals, got {value}');continue
                if name=='workGroups' and not all(re.fullmatch(r'[+-]?\d+',x) for x in arguments):
                    errors.append(f'{file.name}: integer dispatch directive got {value}')
                if any(float(x)<=0 for x in arguments): errors.append(f'{file.name}: non-positive dispatch {value}')
                directive_counts[name]+=1
        for sizes in re.findall(r'layout\s*\(([^)]*local_size_[^)]*)\)',text):
            local=[int(x) for x in re.findall(r'local_size_[xyz]\s*=\s*(\d+)',sizes)]
            if local and math.prod(local)>caps['GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS']:
                errors.append(f'{file.name}: local invocation product {math.prod(local)} exceeds hardware')
    if sum(directive_counts.values())!=expected_compute:
        errors.append(f'Parsed literal dispatch metadata {directive_counts} does not cover {expected_compute} compute roots')
    cinematic=int(options.get('CINEMATIC_MODE',0))
    cine_expected={'VX_VOL_SIZE':2,'VOXEL_DETAIL_AMOUNT':8,'SDF_UPDATE_INTERVAL':1,
                   'GI_STRENGTH':2,'GI_SAMPLE_COUNT':13,'BLOCKLIGHT_RESOLUTION':1,
                   'MAX_TRACE_COUNT':128,'RT_STEPS':100,'LIGHT_TRACE_LENGTH':100,
                   'SHADOW_QUALITY':5,'DETAIL_QUALITY':3,'CLOUD_QUALITY':3,
                   'LIGHTSHAFT_QUALI_DEFINE':4,'WATER_REFLECT_QUALITY':2,
                   'BLOCK_REFLECT_QUALITY':3,'SSAO_QUALI_DEFINE':3,
                   'ANISOTROPIC_FILTER':16,'POM_QUALITY':512,'POM_DISTANCE':1024,
                   'TAA_MODE':1,'ENTITY_SHADOWS_DEFINE':1}
    if cinematic==1:
        for name,expected in cine_expected.items():
            if effective_values.get(name)!={float(expected)}:
                errors.append(f'Cinematic effective {name}: expected{expected}, actual{effective_values.get(name)}')
        for name in ['PER_PIXEL_LIGHT','LONGER_LIGHT_LISTS','TRANSLUCENT_LIGHT_TINT','GI','BLOCKLIGHT_HIGHLIGHT','ENTITY_VOXELIZATION','PLAYER_VOXELIZATION','FOLIAGE_VOXELIZATION','VOXEL_RT_REFLECTIONS']:
            if boolean_values.get(name)!={1}: errors.append(f'Cinematic force-onbool {name}: actual{boolean_values.get(name)}')
        if boolean_values.get('PER_BLOCK_LIGHT')!={0}: errors.append('Cinematic failed to disable PER_BLOCK_LIGHT')
    camera_fields=['WORLD_BLUR','WB_DOF_I','WB_DOF_FOCUS','BARREL_DISTORTION','BARREL_DISTORTION_MODE','CHROMA_ABERRATION','STREAK_COLOR_MODE','STREAK_THRESHOLD','STREAK_STRETCH','STREAK_INTENSITY','LENSFLARE_MODE','LENSFLARE_I','T_EXPOSURE','T_SATURATION','MOTION_BLURRING_STRENGTH']
    for name in camera_fields:
        if name in original_values and effective_values.get(name)!={float(original_values[name])}:
            errors.append(f'Artistic/camera option unexpectedly altered {name}: original{original_values[name]}, effective{effective_values.get(name)}')
    for name in ['KINO_STREAK','MOTION_BLURRING','WB_CHROMATIC','WB_ANAMORPHIC','WB_FOV_SCALED','BLOOM','COLORGRADING']:
        if name in original_values and boolean_values.get(name)!={int(original_values[name]=='true')}:
            errors.append(f'Camera boolean unexpectedly altered {name}: original{original_values[name]}, effective{boolean_values.get(name)}')
    # Cinematic Quality must not rewrite the user's shadow draw distance, so
    # coverage depends on that stored value: geometry is only voxelized inside
    # the shadow render distance, and the movie volume reaches 128 blocks out.
    # The installed configuration must satisfy it; a synthetic profile only
    # informs, because its preset distance is not the user's stored choice.
    expected_shadow=float(original_values['shadowDistance'])
    if shadow_distances!={expected_shadow}: errors.append(f'shadowDistance wrong: {shadow_distances} expected{expected_shadow}')
    if cinematic==1 and len(volume_sizes)==1:
        half_extent=max(next(iter(volume_sizes)))*.5
        if expected_shadow<half_extent:
            message=(f'shadow draw distance {expected_shadow} < movie voxel half-extent {half_extent}; '
                     'outer movie voxels stay unpopulated')
            (errors if variant.startswith('current') else warnings).append(message)
    if len(volume_sizes)!=1: errors.append(f'Shader volume size inconsistent/absent across compute sources: {sorted(volume_sizes)}')
    else:
        x,y,z=next(iter(volume_sizes))
        ratios={'irradianceCacheI':2,'irradianceSnapshotI':2,'occupancyVolume':2,
                'lightStorage':1,'lightStorageSnapshotI':1,'voxel_img':1,
                'distanceFieldI':4,'voxelCols':4,'voxelLightDataI':4}
        for image in images:
            name=image['name'].removeprefix('image.')
            if name in ratios and image['dimensions']!=[x,y*ratios[name],z]:
                errors.append(f"Shader/property volume mismatch {name}: shader{(x,y,z)} vsimage{image['dimensions']}")
    return {'custom_image_count':len(images),'custom_images':images,'ssbos':buffers,
            'actual_shader_volume_sizes':[list(v) for v in sorted(volume_sizes)],
            'effective_cinematic_numeric_values':{key:sorted(values) for key,values in effective_values.items()},
            'effective_cinematic_boolean_values':{key:sorted(values) for key,values in boolean_values.items()},
            'effective_shadow_distances':sorted(shadow_distances),
            'original_public_defaults_preserved_and_cinematic_menu_checked_by_actual_Iris':True,
            'shader_property_volume_dimensions_agree':not any('volume' in error for error in errors),
            'actual_JCPP_compute_metadata_files':len(metadata_files),'literal_dispatch_directive_counts':directive_counts,
            'estimated_custom_image_and_ssbo_MiB_at_2560x1351':round(total/(1024**2),3),
            'estimate_excludes_main_colortex_shadow_textures_and_driver_overheads':True,
            'errors':errors,'warnings':warnings,'preprocessor':'actual installed Iris/JCPP'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', nargs='+', default=['LOW', 'HIGH', 'ULTRA', 'current'])
    parser.add_argument('--cinematic', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--affected-only', action='store_true')
    parser.add_argument('--iris-options', action='store_true', help='Also run installed Iris option/property integration harness')
    parser.add_argument('--skip-property-audit', action='store_true', help='Skip actual Iris/JCPP metadata audit during an in-flight smoke run')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    ctx = moderngl.create_standalone_context(require=430)
    caps = capability_limits(ctx)
    print('GPU', ctx.info['GL_RENDERER'], '|', ctx.info['GL_VERSION'], flush=True)
    print('Capabilities', json.dumps(caps), flush=True)
    # Native 430 compatibility must compile; don't rewrite versions to fake it.
    probe = ctx.program(vertex_shader='#version 430 compatibility\nvoid main(){gl_Position=ftransform();}',
                        fragment_shader='#version 430 compatibility\nvoid main(){gl_FragData[0]=vec4(1.0);}')
    probe.release()
    print('PASS original GLSL430 compatibility context probe', flush=True)
    profiles = {name.removeprefix('profile-'): opts for name, opts in streak.profiles().items()}
    profiles['current'] = current_options()
    env = environment(ctx)
    check_wrapper_chains()
    start_hash = source_snapshot()
    counts = {'compute': 0, 'raster': 0, 'stages': 0, 'geometry': 0, 'attempted_compute': 0, 'attempted_raster': 0}
    failures = []
    active_limits = {'max_image_uniforms_in_program': 0, 'programs_with_max_images': []}
    property_audits = {}
    property_runner = None if args.skip_property_audit else prepare_property_preprocessor()
    began = time.monotonic()
    for label in args.profiles:
        if label not in profiles:
            raise ValueError('Unknown profile ' + label)
        for cinematic in args.cinematic:
            options = dict(profiles[label], CINEMATIC_MODE=cinematic)
            variant = f'{label}-cinematic{cinematic}'
            actual_stage_root = None
            if property_runner:
                try:
                    audit = audit_properties(*property_runner, variant, options, caps)
                    property_audits[variant] = audit
                    actual_stage_root = CACHE/(variant+'.preprocessed.properties.expanded-stages')
                    print('PROPERTIES', variant, audit['custom_image_count'], 'images', audit['estimated_custom_image_and_ssbo_MiB_at_2560x1351'], 'MiB', flush=True)
                    for error in audit['errors']:
                        failures.append({'variant':variant,'world':'properties','program':'metadata','error':error})
                    for warning in audit.get('warnings',[]):
                        print('NOTE', variant, warning, flush=True)
                except Exception as exc:
                    failures.append({'variant':variant,'world':'properties','program':'preprocessor','error':str(exc)})
                    print('FAIL properties',variant,str(exc),flush=True)
            for world in WORLDS:
                passed = 0
                groups = stage_sets(world, args.affected_only)
                for name, paths in groups.items():
                    kind = 'compute' if name.startswith('compute:') else 'raster'
                    counts['attempted_' + kind] += 1
                    sources = {}
                    try:
                        sources = {stage: shader_source(path, options, env, actual_stage_root) for stage, path in paths.items()}
                        if kind == 'compute':
                            program = ctx.compute_shader(sources['compute_shader'])
                        else:
                            program = ctx.program(**sources)
                        # Successful actual link checks per-stage image, sampler,
                        # SSBO and geometry resource limits on the driver itself.
                        image_count = sum(getattr(program[member], 'array_length', 1) for member in program
                                          if 0x904C <= getattr(program[member], 'gl_type', 0) <= 0x906D)
                        if image_count > active_limits['max_image_uniforms_in_program']:
                            active_limits['max_image_uniforms_in_program'] = image_count
                            active_limits['programs_with_max_images'] = []
                        if image_count == active_limits['max_image_uniforms_in_program'] and image_count:
                            active_limits['programs_with_max_images'].append(f'{variant}/{world}/{name}')
                        assert image_count <= caps['GL_MAX_IMAGE_UNITS'], (name,image_count,caps['GL_MAX_IMAGE_UNITS'])
                        program.release()
                        counts[kind] += 1
                        counts['stages'] += len(sources)
                        counts['geometry'] += int('geometry_shader' in sources)
                        passed += 1
                    except Exception as exc:
                        folder = CACHE / variant / world / name.replace(':', '-')
                        folder.mkdir(parents=True, exist_ok=True)
                        for stage, src in sources.items():
                            (folder / (stage + '.glsl')).write_text(src, encoding='utf-8')
                        (folder / 'error.txt').write_text(str(exc), encoding='utf-8')
                        failure = {'variant': variant, 'world': world, 'program': name,
                                   'paths': [str(p.relative_to(ROOT)) for p in paths.values()],
                                   'error': str(exc), 'saved_sources': str(folder)}
                        failures.append(failure)
                        print('FAIL', variant, world, name, str(exc)[:240].replace('\n',' | '), flush=True)
                print('COMPLETE', variant, world, f'{passed}/{len(groups)} programs', flush=True)
    end_hash = source_snapshot()
    report = {'gpu': ctx.info['GL_RENDERER'], 'version': ctx.info['GL_VERSION'],
              'profiles': args.profiles, 'cinematic_modes': args.cinematic,
              'requested_option_snapshots': {label:profiles[label] for label in args.profiles},
              'affected_only': args.affected_only, 'counts': counts, 'capabilities': caps,
              'active_image_uniform_limits': active_limits, 'property_audits': property_audits, 'failures': failures,
              'snapshot_start': start_hash, 'snapshot_end': end_hash,
              'source_changed_during_run': start_hash != end_hash,
              'elapsed_seconds': round(time.monotonic() - began, 2),
              'notes': ['No compute dispatches or texture/SSBO allocations; compilation/link regression only.',
                        'Original shader versions and algorithm bodies retained; Iris/MC environment constants injected.',
                        'Default audit mode compiles actual installed Iris option-rewritten/IncludeProcessor-expanded stage sources.',
                        'Iris-supplied line and DH engine interface declarations are injected; DH entry points compile with DH module enabled.',
                        'Actual Iris/JCPP evaluates properties and compute metadata; no global custom-image count conflation with per-program image units.',
                        'Entry points compiled even if program.enabled would disable them for selected profile.',
                        'Source changes during a run require a settled rerun.']}
    output = CACHE / 'latest-report.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('SUMMARY', json.dumps(counts), 'failures', len(failures), 'changed', start_hash != end_hash, flush=True)
    print('REPORT', output, flush=True)
    ctx.release()
    if args.iris_options:
        subprocess.run([sys.executable, str(ROOT / 'tools/test_iris_streak_options.py')], check=True)
    return int(bool(failures) or start_hash != end_hash)


if __name__ == '__main__':
    sys.exit(main())
