"""Standalone NVIDIA GPU regression tests; no game, installs, or profile changes.
Run: python tools/test_kino_streak.py
CPU reference models exact sky-depth masking BEFORE interpolation, fused
threshold-before-horizontal-interpolation prefilter, six-tap downsampling,
reconstruction and rgba16f rounding at every stage. Compute placement is
composite3 before the real world blur; composite5 only composes the atlas.
"""
from pathlib import Path
import re
import sys
import numpy as np
import moderngl

ROOT = Path(__file__).resolve().parents[1] / 'shaders'
WORLDS = ('world0', 'world-1', 'world1')


def expand(path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError(f'include cycle: {stack + (path,)}')
    text = path.read_text(encoding='utf-8')
    def include(m):
        name = m.group(1)
        target = ROOT / name.lstrip('/') if name.startswith('/') else path.parent / name
        return expand(target, stack + (path,))
    return re.sub(r'^\s*#include\s+"([^"]+)"[^\n]*', include, text, flags=re.M)


def source(path, options=None):
    text = expand(path)
    # Emulate Iris option rewriting, including commented-out boolean options.
    for name, value in (options or {}).items():
        pattern = rf'^(\s*)(?://\s*)?#define\s+{re.escape(name)}\b[^\n]*'
        replacement = '// disabled ' + name if value is False else '#define ' + name + ('' if value is True else ' ' + str(value))
        text, count = re.subn(pattern, lambda m: replacement, text, flags=re.M)
        if not count:
            # Iris profiles also contain options declared in unrelated passes.
            text = re.sub(rf'(const\s+\w+\s+{re.escape(name)}\s*=\s*)[^;]+;',
                          lambda m: m.group(1) + str(value) + ';', text)
    pos = text.index('\n') + 1
    environment = '#define IS_IRIS\n#define IRIS_VERSION 11107\n#define MC_VERSION 260300\n'
    if (options or {}).get('DISTANT_HORIZONS'):
        environment += '#define DISTANT_HORIZONS\n'
    return text[:pos] + environment + text[pos:]


def widths(w):
    result = [max(1, w // 2)]
    while len(result) < 10 and result[-1] > 16:
        result.append(max(1, w >> (len(result) + 1)))
    return result


def half(a):
    return a.astype(np.float16).astype(np.float64)


def linear_x(a, u):
    x = np.clip(np.asarray(u) * a.shape[1] - .5, 0, a.shape[1] - 1)
    lo = np.floor(x).astype(int)
    f = (x - lo)[None, :, None]
    return a[:, lo] * (1 - f) + a[:, np.minimum(lo + 1, a.shape[1] - 1)] * f


def reference(src, threshold=1., stretch=.75, depth=None, dh_depth=None):
    h, w, _ = src.shape
    ah = max(1, h // 2)
    # Mask exact source texels BEFORE any vertical or horizontal interpolation.
    # Thus even an adjacent extremely bright sky texel cannot contaminate a
    # below-threshold world texel and become a false streak source.
    if depth is not None:
        world = depth < 1.0
        if dh_depth is not None:
            # DH may have a different render resolution. Match normalized pixel
            # centers with nearest depth lookup before classifying each source.
            dh_h,dh_w = dh_depth.shape
            xx = np.minimum(((np.arange(w)+.5)*dh_w/w).astype(int),dh_w-1)
            yy = np.minimum(((np.arange(h)+.5)*dh_h/h).astype(int),dh_h-1)
            world |= dh_depth[yy[:,None],xx[None,:]] < 1.0
        src = src.copy()
        src[:,:,:3] *= world[:,:,None]
    # Vertical bilinear samples at half-height pixel centers +/- .75 source pixels.
    y = (np.arange(ah) + .5) * h / ah - .5
    samples = []
    for delta in (-.75, .75):
        yy = np.clip(y + delta, 0, h - 1)
        lo = np.floor(yy).astype(int)
        f = (yy - lo)[:, None, None]
        samples.append(src[lo, :, :3] * (1-f) + src[np.minimum(lo+1, h-1), :, :3] * f)
    pre = (samples[0] + samples[1]) * .5
    bright = np.max(pre, axis=2, keepdims=True)
    pre *= np.maximum(0, bright-threshold) / np.maximum(bright, 1e-5)
    down = []
    for width in widths(w):
        previous = pre if not down else down[-1]
        u = (np.arange(width) + .5) / width
        c = sum(linear_x(previous, u + tap * 1.25 / previous.shape[1]) for tap in (-5,-3,-1,1,3,5)) / 6
        down.append(half(c))
    up = [None] * len(down)
    for i in reversed(range(len(down))):
        c = down[i]
        if i + 1 < len(down):
            larger = linear_x(up[i+1], (np.arange(c.shape[1])+.5)/c.shape[1])
            c = c * (1-stretch) + larger * stretch
        up[i] = half(c)
    return down, up


def dispatch(ctx, programs, src, enabled=True, threshold=1., stretch=.75,
             atlas_height=None, depth=None, dh_depth=None):
    h, w, _ = src.shape
    depth = np.full((h,w),.9,dtype='f4') if depth is None else np.asarray(depth,dtype='f4')
    assert depth.shape == (h,w)
    depth_texture = ctx.texture((w,h),1,depth.tobytes(),dtype='f4')
    depth_texture.filter = (moderngl.NEAREST,moderngl.NEAREST)
    depth_texture.repeat_x = depth_texture.repeat_y = False
    depth_texture.use(1)
    dh_texture = None
    if dh_depth is not None:
        dh_depth = np.asarray(dh_depth,dtype='f4')
        assert dh_depth.ndim == 2 and min(dh_depth.shape) > 0
        dh_h,dh_w = dh_depth.shape
        dh_texture = ctx.texture((dh_w,dh_h),1,dh_depth.tobytes(),dtype='f4')
        dh_texture.filter = (moderngl.NEAREST,moderngl.NEAREST)
        dh_texture.repeat_x = dh_texture.repeat_y = False
        dh_texture.use(2)
    ah = max(1, h//2) if atlas_height is None else atlas_height
    tex = ctx.texture((w,h), 4, src.astype('f4').tobytes(), dtype='f4')
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = False
    tex.use(0)
    sentinel = np.full((ah,w,4), -17, dtype='f2')
    atlases = [ctx.texture((w,ah),4,sentinel.tobytes(),dtype='f2') for _ in range(2)]
    for program in programs:
        for name, binding in [('colortex0',0),('depthtex0',1),('dhDepthTex',2),
                              ('streakDownImage',0),('streakUpImage',1)]:
            if name in program:
                program[name].value = binding
        for binding, atlas in enumerate(atlases):
            atlas.bind_to_image(binding, read=True, write=True)
        # Deliberately over-dispatch: bounds guards must protect inactive tiles.
        program.run(group_x=(w+7)//8, group_y=(ah+7)//8)
        ctx.memory_barrier()
    arrays = [np.frombuffer(t.read(), dtype='f2').reshape(ah,w,4).astype('f8') for t in atlases]
    if enabled:
        refs = reference(src, threshold, stretch, depth, dh_depth)
        for actual, levels in zip(arrays,refs):
            offset = 0
            for level in levels:
                width = level.shape[1]
                np.testing.assert_allclose(actual[:,offset:offset+width,:3], level, atol=.012, rtol=.003)
                np.testing.assert_array_equal(actual[:,offset:offset+width,3], 1)
                offset += width
            np.testing.assert_array_equal(actual[:,offset:], -17)
    else:
        for actual in arrays:
            np.testing.assert_array_equal(actual, -17)
    for t in [tex, depth_texture, *atlases]:
        t.release()
    if dh_texture is not None:
        dh_texture.release()
    return arrays


def profiles():
    result = {}
    for name, settings in re.findall(r'^\s*profile\.(\w+)\s*=([^\n]+)', (ROOT/'shaders.properties').read_text(), re.M):
        options = {}
        for token in settings.split():
            if '=' in token:
                key, val = token.split('=',1)
                options[key] = val
            elif token.startswith('!'):
                options[token[1:]] = False
            else:
                options[token] = True
        result['profile-'+name] = options
    return result


def test_composition(ctx):
    # Compile the real helper, not a translated replacement algorithm.
    common = (ROOT/'lib/common.glsl').read_text()
    names = ('STREAK_COLOR_MODE','STREAK_INTENSITY','STREAK_TINT_R','STREAK_TINT_G','STREAK_TINT_B')
    defines = '\n'.join(re.search(r'^\s*(#define\s+' + name + r'\b[^\n]*)',common,re.M).group(1) for name in names)
    vert = '''#version 330
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}'''
    frag = '#version 330\n' + defines + '\n' + expand(ROOT/'lib/misc/kinoStreakComposite.glsl')
    frag += '\nuniform ivec2 sceneSize; in vec2 uv; out vec4 color; void main() { color=vec4(GetKinoStreak(uv,sceneSize),1.0); }\n'
    program = ctx.program(vertex_shader=vert,fragment_shader=frag)
    vao = ctx.vertex_array(program, [])
    valid_cases = 0
    for w,h in [(32,16),(67,9),(3,5)]:
        program['sceneSize'].value = (w,h)
        aw, ah = w, max(1,h//2)
        for constant in (True, False):
            data = np.full((ah,aw,4),1234,dtype='f2')
            sw = max(1,aw//2)
            data[:,:sw,:3] = 6 if constant else np.random.default_rng(21).uniform(0,12,(ah,sw,3))
            atlas = ctx.texture((aw,ah),4,data.tobytes(),dtype='f2')
            # Repeat/nearest deliberately adverse: helper must do its own clamping/filtering.
            atlas.filter = (moderngl.NEAREST,moderngl.NEAREST)
            atlas.repeat_x = atlas.repeat_y = True
            atlas.use(0)
            program['streakUpSampler'].value = 0
            target = ctx.texture((w,h),4,dtype='f4')
            framebuffer = ctx.framebuffer(color_attachments=[target])
            framebuffer.use()
            ctx.viewport = (0,0,w,h)
            vao.render(mode=moderngl.TRIANGLES,vertices=3)
            actual = np.frombuffer(target.read(),dtype='f4').reshape(h,w,4)[:,:,:3]
            tile = data[:,:sw,:3].astype('f8')
            horizontal = linear_x(tile,(np.arange(w)+.5)/w)
            yy = np.clip((np.arange(h)+.5)*ah/h-.5,0,ah-1)
            lo = np.floor(yy).astype(int)
            f = (yy-lo)[:,None,None]
            expected = (horizontal[lo]*(1-f)+horizontal[np.minimum(lo+1,ah-1)]*f)*np.array([.55,.55,1.])*1.5
            # Rasterized float32 UV interpolation differs from float64 CPU by ~3e-5.
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=5e-5)
            if constant:
                np.testing.assert_allclose(actual,np.broadcast_to([4.95,4.95,9.],actual.shape),atol=2e-6)
            framebuffer.release(); target.release(); atlas.release()
            valid_cases += 1
    vao.release(); program.release()

    # Reproduce Iris's missing custom-sampler registration failure exactly:
    # a freshly linked sampler is left unset (OpenGL default texture unit 0),
    # and unit 0 contains a FULL-height HDR scene rather than the atlas.
    # The helper must reject it before sampling, not overlay a stretched scene.
    def fragment_with_options(options):
        result = frag
        for name,value in options.items():
            result,count = re.subn(r'^#define\s+'+re.escape(name)+r'\b[^\n]*',
                                  '#define '+name+' '+str(value),result,flags=re.M)
            assert count == 1, (name,count)
        return result

    color_cases = 0
    # Mode zero must preserve source RGB and user tint exactly; mode one must
    # discard source hue AFTER bilinear sampling and ignore every tint slider.
    for mode in (0,1):
        for tint in ((.55,.55,1.),(.9,.1,.2),(0.,0.,0.)):
            options = dict(zip(('STREAK_TINT_R','STREAK_TINT_G','STREAK_TINT_B'),tint))
            options['STREAK_COLOR_MODE'] = mode
            color_program = ctx.program(vertex_shader=vert,fragment_shader=fragment_with_options(options))
            color_vao = ctx.vertex_array(color_program,[])
            w,h = 7,5
            aw,ah = w,h//2
            sw = w//2
            color_program['sceneSize'].value = (w,h)
            color_program['streakUpSampler'].value = 0
            sources = {'red':(6,0,0),'yellow':(6,6,0),'blue':(0,0,6),
                       'white':(6,6,6),'negative':(-6,-2,-1)}
            mixed = np.array([[[6,0,0],[0,0,6],[0,6,0]],
                              [[0,6,6],[6,0,6],[6,6,0]]],dtype='f2')
            fixtures = [(label,np.broadcast_to(rgb,(ah,sw,3))) for label,rgb in sources.items()]
            fixtures.append(('mixed-bilinear',mixed))
            for label,tile in fixtures:
                data = np.full((ah,aw,4),1234,dtype='f2')
                data[:,:sw,:3] = tile
                atlas = ctx.texture((aw,ah),4,data.tobytes(),dtype='f2')
                atlas.filter = (moderngl.NEAREST,moderngl.NEAREST)
                atlas.repeat_x = atlas.repeat_y = True
                atlas.use(0)
                target = ctx.texture((w,h),4,dtype='f4')
                framebuffer = ctx.framebuffer(color_attachments=[target])
                framebuffer.use(); ctx.viewport=(0,0,w,h)
                color_vao.render(mode=moderngl.TRIANGLES,vertices=3)
                actual = np.frombuffer(target.read(),dtype='f4').reshape(h,w,4)[:,:,:3]
                horizontal = linear_x(data[:,:sw,:3].astype('f8'),(np.arange(w)+.5)/w)
                yy = np.clip((np.arange(h)+.5)*ah/h-.5,0,ah-1)
                lo = np.floor(yy).astype(int); f=(yy-lo)[:,None,None]
                sampled = horizontal[lo]*(1-f)+horizontal[np.minimum(lo+1,ah-1)]*f
                if mode == 0:
                    expected = sampled*np.array(tint)*1.5
                else:
                    expected = np.maximum(0,np.max(sampled,axis=2,keepdims=True))*np.array([.25,.55,1.])*1.5
                np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=5e-5,
                                           err_msg=f'mode {mode}, tint {tint}, source {label}')
                if mode == 1 and label in ('red','yellow','blue','white'):
                    np.testing.assert_allclose(actual,np.broadcast_to([2.25,4.95,9.],actual.shape),atol=2e-6)
                if mode == 1 and label == 'negative':
                    np.testing.assert_array_equal(actual,0)
                framebuffer.release(); target.release(); atlas.release()
                color_cases += 1
            color_vao.release(); color_program.release()

    fallback_cases = 0
    for mode in (0,1):
        fallback_program = ctx.program(vertex_shader=vert,fragment_shader=fragment_with_options({'STREAK_COLOR_MODE':mode}))
        assert fallback_program['streakUpSampler'].value == 0
        fallback_vao = ctx.vertex_array(fallback_program, [])
        for w,h in [(32,16),(67,9),(3,5)]:
            fallback_program['sceneSize'].value = (w,h)
            scene_data = np.random.default_rng(91).uniform(2,32,(h,w,4)).astype('f4')
            scene = ctx.texture((w,h),4,scene_data.tobytes(),dtype='f4')
            scene.filter = (moderngl.LINEAR,moderngl.LINEAR)
            scene.use(0)
            # Intentionally NEVER assign fallback_program['streakUpSampler'].
            target = ctx.texture((w,h),4,dtype='f4')
            framebuffer = ctx.framebuffer(color_attachments=[target])
            framebuffer.use()
            ctx.viewport = (0,0,w,h)
            fallback_vao.render(mode=moderngl.TRIANGLES,vertices=3)
            actual = np.frombuffer(target.read(),dtype='f4').reshape(h,w,4)
            np.testing.assert_array_equal(actual[:,:,:3],0)
            np.testing.assert_array_equal(actual[:,:,3],1)
            assert fallback_program['streakUpSampler'].value == 0
            framebuffer.release(); target.release(); scene.release()
            fallback_cases += 1
        fallback_vao.release(); fallback_program.release()
    print(f'PASS actual composite helper: {valid_cases} original default-atlas cases, '
          f'{color_cases} current/classic-blue source/tint/bilinear cases, '
          f'{fallback_cases} unset-sampler rejection cases in both modes (exact zero)')
    return valid_cases + color_cases + fallback_cases


def assert_pipeline_structure():
    for world in WORLDS:
        paths = sorted((ROOT/world).glob('composite3*.csh'))
        assert len(paths) == 20, (world, len(paths))
        assert not list((ROOT/world).glob('composite5*.csh')), 'stale post-blur streak compute remains'
        for p in paths:
            text = p.read_text()
            assert '#define COMPOSITE3' in text and '/program/kinoStreak.glsl' in text, p
    c3 = (ROOT/'program/composite3.glsl').read_text()
    main = c3.split('//Program//',1)[1]
    assert main.index('DoWorldBlur(color,') < main.index('gl_FragData[0]'), 'world blur must stay in composite3 raster'
    c4 = (ROOT/'program/composite4.glsl').read_text()
    c5 = (ROOT/'program/composite5.glsl').read_text()
    assert 'BloomTile(' in c4 and 'DRAWBUFFERS:3' in c4
    assert c5.index('color += GetKinoStreak(') < c5.index('DoBSLTonemap(color)')
    print('PASS structure: 20 composite3 computes per world before composite3 world blur; composite4 bloom, composite5 HDR composition')


def compile_compute(ctx, world, options=None):
    return [ctx.compute_shader(source(p,options)) for p in sorted((ROOT/world).glob('composite3*.csh'))]


def active_values(array,w):
    return array[:,:sum(widths(w)),:3]


def test_depth_sources(ctx, programs, world):
    w,h = 513,17
    sky = np.ones((h,w),dtype='f4')
    huge = np.full((h,w,4),60000,dtype='f4')
    for a in dispatch(ctx,programs,huge,depth=sky):
        np.testing.assert_array_equal(active_values(a,w),0)
    # Bright sky touches world texels on both vertical and horizontal boundaries.
    depth = np.full((h,w),.9,dtype='f4')
    depth[::2,:w//2] = 1
    depth[:4,:] = 1
    src = np.full((h,w,4),.5,dtype='f4')
    src[depth>=1,:3] = 60000
    for a in dispatch(ctx,programs,src,depth=depth):
        np.testing.assert_array_equal(active_values(a,w),0)
    # A real emitter adjacent to sky must remain; replacing sky with zero RGB
    # must yield precisely the same atlas as filtering exact depth-masked texels.
    src[8:11,255:258,:3] = (64,32,16)
    depth[8:11,255:258] = .9
    masked = src.copy(); masked[depth>=1,:3] = 0
    actual = dispatch(ctx,programs,src,depth=depth)
    clean = dispatch(ctx,programs,masked,depth=depth)
    for a,b in zip(actual,clean):
        np.testing.assert_array_equal(a,b)
    assert np.max(active_values(actual[1],w)) > 0, 'world emitter disappeared next to sky'
    print('PASS exact-depth source mask: all-sky huge HDR, mixed-boundary no bleed, retained world emitter',world)


def test_dh_sources(ctx,world):
    programs = compile_compute(ctx,world,{'DISTANT_HORIZONS':True})
    try:
        w,h = 129,13
        depth = np.ones((h,w),dtype='f4')
        dh = np.ones((h,w),dtype='f4')
        src = np.full((h,w,4),60000,dtype='f4')
        for a in dispatch(ctx,programs,src,depth=depth,dh_depth=dh):
            np.testing.assert_array_equal(active_values(a,w),0)
        # Vanilla depth remains sky, but DH contains distant emitter geometry.
        dh[5:8,62:66] = .8
        src[5:8,62:66,:3] = (64,32,16)
        result = dispatch(ctx,programs,src,depth=depth,dh_depth=dh)
        assert np.max(active_values(result[1],w)) > 0, 'DH distant geometry excluded as sky'
        clean = src.copy(); clean[dh>=1,:3] = 0
        other = dispatch(ctx,programs,clean,depth=depth,dh_depth=dh)
        for a,b in zip(result,other): np.testing.assert_array_equal(a,b)
        # Exercise normalized mapping when the DH depth texture is lower-res.
        low_dh = np.ones((7,65),dtype='f4'); low_dh[2:5,30:34] = .8
        low_src = np.full((h,w,4),8,dtype='f4')
        low_result = dispatch(ctx,programs,low_src,depth=depth,dh_depth=low_dh)
        assert np.max(active_values(low_result[1],w)) > 0
        print('PASS DH source mask: true sky excluded, distant geometry retained, lower-resolution DH mapping',world)
    finally:
        for p in programs: p.release()
    return len(programs)


def test_worldblur_independence(ctx,world):
    w,h = 257,33
    src = np.random.default_rng(911).uniform(0,4,(h,w,4)).astype('f4')
    src[8:12,91:94,:3] = (64,32,16)
    depth = np.full((h,w),.9,dtype='f4'); depth[:6] = 1
    src[:6,:,:3] = 60000
    baseline = None
    cases = 0
    for blur in (0,1,2):
        for chromatic in (False,True):
            options = {'WORLD_BLUR':blur,'WB_CHROMATIC':chromatic,'WB_ANAMORPHIC':True}
            programs = compile_compute(ctx,world,options)
            try:
                actual = dispatch(ctx,programs,src,depth=depth)
                if baseline is None: baseline = actual
                else:
                    for a,b in zip(actual,baseline): np.testing.assert_array_equal(a,b)
                cases += 1
            finally:
                for p in programs: p.release()
    print('PASS raw-source atlas bit-identical across WORLD_BLUR=0/1/2, chromatic off/on, anamorphic enabled',world)
    return cases * 20


def extract_function(text,name):
    start = text.index('void '+name+'(')
    begin = text.index('{',start)
    depth = 1; end = begin+1
    while depth:
        depth += (text[end]=='{') - (text[end]=='}')
        end += 1
    return text[start:end]


def test_actual_worldblur_order(ctx):
    # Render the actual pack's DoWorldBlur function (not a hand-coded RGB shift),
    # then feed the post-blur result into the streak filter to reproduce the old
    # placement. The new pre-composite3 placement samples the untouched source.
    c3 = (ROOT/'program/composite3.glsl').read_text()
    offsets = re.search(r'vec2 dofOffsets\[18\].*?\);',c3,re.S).group(0)
    function = extract_function(c3,'DoWorldBlur')
    vert = '''#version 330
out vec2 texCoord;
void main() { vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2); texCoord=p; gl_Position=vec4(p*2-1,0,1); }
'''
    frag = '''#version 330
#define WORLD_BLUR 2
#define WB_DOF_FOCUS 0
#define WB_DOF_I 48.0
#define WB_CHROMATIC
#define WB_ANAMORPHIC
#define texture2DLod textureLod
uniform sampler2D colortex0;
uniform float viewWidth,viewHeight,aspectRatio,centerDepthSmooth;
in vec2 texCoord; out vec4 outputColor;
'''+offsets+'\n'+function+'''
void main() { vec3 color=texture(colortex0,texCoord).rgb; DoWorldBlur(color,.95,50.0); outputColor=vec4(color,1); }
'''
    program = ctx.program(vertex_shader=vert,fragment_shader=frag)
    vao = ctx.vertex_array(program,[])
    w,h = 512,64
    src = np.zeros((h,w,4),dtype='f4'); src[30:35,382:387,:3] = 64
    tex = ctx.texture((w,h),4,src.tobytes(),dtype='f4')
    tex.filter = (moderngl.LINEAR,moderngl.LINEAR); tex.repeat_x=tex.repeat_y=False; tex.build_mipmaps()
    tex.use(0); program['colortex0'].value=0
    for name,value in [('viewWidth',w),('viewHeight',h),('aspectRatio',w/h),('centerDepthSmooth',0.)]:
        program[name].value=value
    target = ctx.texture((w,h),4,dtype='f4'); framebuffer=ctx.framebuffer(color_attachments=[target])
    framebuffer.use(); ctx.viewport=(0,0,w,h); vao.render(mode=moderngl.TRIANGLES,vertices=3)
    blurred = np.frombuffer(target.read(),dtype='f4').reshape(h,w,4).copy()
    assert np.max(np.abs(blurred[:,:,0]-blurred[:,:,2])) > 1, 'actual chromatic worldblur did not shift channels'
    programs = compile_compute(ctx,'world0',{'WORLD_BLUR':2,'WB_CHROMATIC':True,'WB_ANAMORPHIC':True})
    try:
        raw = dispatch(ctx,programs,src)[1][:,:w//2,:3]
        old = dispatch(ctx,programs,blurred)[1][:,:w//2,:3]
        np.testing.assert_array_equal(raw[:,:,0],raw[:,:,2])
        assert np.max(np.abs(old[:,:,0]-old[:,:,2])) > .01, 'old post-blur placement did not reproduce RGB-split streak'
        again = dispatch(ctx,programs,src)[1][:,:w//2,:3]
        np.testing.assert_array_equal(raw,again)
    finally:
        for p in programs: p.release()
        framebuffer.release(); target.release(); tex.release(); vao.release(); program.release()
    print('PASS actual DoWorldBlur two-pass regression: postblur source RGB-splits streak; unchanged preblur source remains achromatic')
    return len(programs)


def main():
    assert_pipeline_structure()
    ctx = moderngl.create_standalone_context(require=430)
    print('GPU:',ctx.info['GL_RENDERER'], '|',ctx.info['GL_VERSION'])
    variants = {'default':{},'bloom-off':{'BLOOM':False},'taa-off':{'TAA_MODE':0},
                'classic-blue':{'STREAK_COLOR_MODE':1},
                'classic-blue-worldblur2-chromatic':{'STREAK_COLOR_MODE':1,'WORLD_BLUR':2,
                                                     'WB_CHROMATIC':True,'WB_ANAMORPHIC':True},
                'worldblur1':{'WORLD_BLUR':1},
                'worldblur2':{'WORLD_BLUR':2},
                'worldblur1-chromatic':{'WORLD_BLUR':1,'WB_CHROMATIC':True,'WB_ANAMORPHIC':True},
                'worldblur2-chromatic':{'WORLD_BLUR':2,'WB_CHROMATIC':True,'WB_ANAMORPHIC':True},
                'worldblur0-chromatic':{'WORLD_BLUR':0,'WB_CHROMATIC':True,'WB_ANAMORPHIC':True},
                'live-user-combo':{'WORLD_BLUR':2,'WB_CHROMATIC':True,'WB_ANAMORPHIC':True,
                                   'WB_DOF_I':'48.0','MOTION_BLURRING':True,
                                   'MOTION_BLURRING_STRENGTH':'0.65','CHROMA_ABERRATION':1,
                                   'STREAK_THRESHOLD':'2.50'},
                'motion-blur':{'MOTION_BLURRING':True},
                'streak-off':{'KINO_STREAK':False},'intensity0':{'STREAK_INTENSITY':'0'},
                'threshold0-stretch1':{'STREAK_THRESHOLD':'0.00','STREAK_STRETCH':'1.00'},
                **profiles()}
    failures = []
    compiled = 0
    composite_linked = 0
    composition_cases = 0
    rng = np.random.default_rng(144)
    for label, options in variants.items():
        for world in WORLDS:
            programs = []
            try:
                programs = compile_compute(ctx,world,options)
                compiled += len(programs)
                # Every variant/world is dispatched; default gets property tests below.
                src = rng.uniform(0,12,(9,67,4)).astype('f4')
                dispatch(ctx,programs,src, label not in ('streak-off','intensity0'),
                         float(options.get('STREAK_THRESHOLD',1)),float(options.get('STREAK_STRETCH',.75)))
                if label == 'default':
                    test_depth_sources(ctx,programs,world)
                    # Incorrect full-height custom images must remain untouched.
                    # This reproduces an image-binding/dimension failure without
                    # allowing any dispatch to smear scene pixels into an atlas.
                    dispatch(ctx,programs,src,enabled=False,atlas_height=src.shape[0])
                    print('PASS compute mismatched full-height images rejected',world)
                    for w,h in [(1,1),(3,5),(32,16),(129,33),(513,17),(1024,18),(16385,3)]:
                        for value in (0., .5, 8.):
                            src = np.full((h,w,4),value,dtype='f4')
                            arrays = dispatch(ctx,programs,src)
                            expected = max(0,value-1)
                            for a in arrays:
                                np.testing.assert_allclose(a[:,:max(1,w//2),:3],expected,atol=.02)
                    src = np.zeros((18,1024,4),dtype='f4')
                    src[:,512,:3] = (64,32,16)
                    a = dispatch(ctx,programs,src)[1][:,:512,:3]
                    assert np.count_nonzero(a[0,:,0]) > 4, 'impulse did not spread horizontally'
                    src[:] = 0
                    src[:,0,:3] = 64
                    a = dispatch(ctx,programs,src)[0][:,:512,:3]
                    assert np.max(a[:,-20:]) == 0, 'first filter wraps opposite boundary'
                    # Nonlinear-order regression: bright and dark neighbors near threshold.
                    src = rng.uniform(0,2,(13,131,4)).astype('f4')
                    dispatch(ctx,programs,src)
                print('PASS compute',label,world)
            except Exception as exc:
                failures.append(f'compute {label} {world}: {exc}')
                print('FAIL', failures[-1])
            finally:
                for p in programs:
                    p.release()
            for pass_name in ('composite3','composite5'):
                try:
                    p = ctx.program(vertex_shader=source(ROOT/world/(pass_name+'.vsh'),options),
                                    fragment_shader=source(ROOT/world/(pass_name+'.fsh'),options))
                    p.release()
                    composite_linked += 1
                    print('PASS composite original #version 130',pass_name,label,world)
                except Exception as exc:
                    failures.append(f'{pass_name} {label} {world}: {exc}')
                    print('FAIL',failures[-1])
    for world in WORLDS:
        try:
            compiled += test_dh_sources(ctx,world)
            compiled += test_worldblur_independence(ctx,world)
        except Exception as exc:
            failures.append(f'depth/blur regression {world}: {exc}')
            print('FAIL',failures[-1])
    try:
        compiled += test_actual_worldblur_order(ctx)
    except Exception as exc:
        failures.append('actual worldblur two-pass: ' + str(exc))
        print('FAIL',failures[-1])
    try:
        composition_cases = test_composition(ctx)
    except Exception as exc:
        failures.append('composition numerical: ' + str(exc))
        print('FAIL',failures[-1])
    print(f'Compiled {compiled} compute wrappers across {len(variants)} variants; '
          f'linked {composite_linked} composite pairs; {composition_cases} helper numerical cases; '
          f'{len(failures)} failures')
    ctx.release()
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
