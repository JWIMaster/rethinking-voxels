// Regression for the actual installed Iris option scanner/property preprocessor.
// No Minecraft classes, manual GL bindings, or replacement parsers are used.
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.lang.reflect.Constructor;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.regex.Pattern;
import net.irisshaders.iris.shaderpack.include.FileNode;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import net.irisshaders.iris.shaderpack.include.AbsolutePackPath;
import net.irisshaders.iris.shaderpack.include.IncludeGraph;
import net.irisshaders.iris.shaderpack.option.ShaderPackOptions;
import net.irisshaders.iris.shaderpack.preprocessor.PropertiesPreprocessor;

public class TestIrisStreakOptions {
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }

    private static ShaderPackOptions options(Path root, List<String> roots, Map<String,String> changes) throws Exception {
        return options(root, roots, changes, false);
    }

    private static ShaderPackOptions options(Path root, List<String> roots, Map<String,String> changes, boolean removeActivation) throws Exception {
        // The public IncludeGraph loader checks Iris.getIrisConfig(), requiring
        // a live Minecraft runtime. Bypass ONLY that unrelated debug-IO check:
        // actual Iris FileNode still parses includes; actual ShaderPackOptions,
        // OptionAnnotatedSource and PropertiesPreprocessor are untouched.
        var nodes = new HashMap<AbsolutePackPath,FileNode>();
        var queue = new ArrayDeque<AbsolutePackPath>();
        for (String path : roots) queue.add(AbsolutePackPath.fromAbsolutePath(path));
        while (!queue.isEmpty()) {
            var path = queue.removeFirst();
            if (nodes.containsKey(path)) continue;
            String source = Files.readString(path.resolved(root));
            if (removeActivation) source = source.replaceAll("(?m)^\\s*#ifdef KINO_STREAK\\s*\\n\\s*#endif\\s*\\n", "");
            var node = new FileNode(path, ImmutableList.copyOf(source.split("\\R")));
            nodes.put(path, node);
            queue.addAll(node.getIncludes().values());
        }
        Constructor<IncludeGraph> ctor = IncludeGraph.class.getDeclaredConstructor(ImmutableMap.class, ImmutableMap.class);
        ctor.setAccessible(true);
        var graph = ctor.newInstance(ImmutableMap.copyOf(nodes), ImmutableMap.of());
        return new ShaderPackOptions(graph, changes);
    }

    private static Properties preprocess(String properties, ShaderPackOptions options) throws Exception {
        String result = PropertiesPreprocessor.preprocessSource(properties, options, List.of());
        var parsed = new Properties();
        parsed.load(new StringReader(result));
        return parsed;
    }

    private static void checkImages(Properties parsed, boolean expected, String name) {
        check(parsed.containsKey("image.streakDownImage") == expected, name + " down image declaration");
        check(parsed.containsKey("image.streakUpImage") == expected, name + " up image declaration");
        if (expected) {
            check(parsed.getProperty("image.streakDownImage").startsWith("streakDownSampler RGBA RGBA16F FLOAT false true 1.0 0.5"), "down sampler/image mapping");
            check(parsed.getProperty("image.streakUpImage").startsWith("streakUpSampler RGBA RGBA16F FLOAT false true 1.0 0.5"), "up sampler/image mapping");
        }
    }

    public static void main(String[] args) throws Exception {
        Path shaders = Path.of(args[0]);
        Path fixture = Path.of(args[1]);
        Files.createDirectories(fixture);
        // Frozen reproducer of the broken guard. The GLSL define exists but
        // the actual Iris scanner does not count #if defined as a reference.
        Files.writeString(fixture.resolve("original.glsl"), """
            #define KINO_STREAK
            #define STREAK_INTENSITY 30 //[0 5 10 15 20 25 30 40 50 60 75 100]
            #if defined KINO_STREAK && STREAK_INTENSITY > 0
            // original enabled shader code
            #endif
            """);
        String properties = Files.readString(shaders.resolve("shaders.properties"));
        var original = options(fixture, List.of("/original.glsl"), Map.of());
        check(!original.getOptionSet().getBooleanOptions().containsKey("KINO_STREAK"), "Original combined guard must reproduce missing boolean option");
        check(original.getOptionSet().getStringOptions().containsKey("STREAK_INTENSITY"), "Intensity must still be recognized");
        checkImages(preprocess(properties, original), false, "broken original");
        System.out.println("PASS actual Iris reproducer: combined #if guard loses KINO_STREAK and both atlas declarations");

        List<String> roots = List.of("/world0/composite5.fsh", "/world0/composite3.csh",
            "/world-1/composite5.fsh", "/world-1/composite3.csh",
            "/world1/composite5.fsh", "/world1/composite3.csh");
        var realBroken = options(shaders, roots, Map.of(), true);
        check(!realBroken.getOptionSet().getBooleanOptions().containsKey("KINO_STREAK"), "Removing activation from actual shader graph must reproduce missing KINO_STREAK");
        checkImages(preprocess(properties, realBroken), false, "actual source minus activation");
        System.out.println("PASS actual Iris project graph minus activation: missing boolean and omitted atlas declarations reproduced in memory");

        var fixed = options(shaders, roots, Map.of());
        check(fixed.getOptionSet().getBooleanOptions().containsKey("KINO_STREAK"), "Fixed shader sources must expose boolean KINO_STREAK");
        check(fixed.getOptionValues().getBooleanValueOrDefault("KINO_STREAK"), "Default streak toggle must be true");
        checkImages(preprocess(properties, fixed), true, "fixed default");
        System.out.println("PASS actual Iris fixed source: recognized default-ON boolean and correct down/up image sampler mappings");

        var colorOption = fixed.getOptionSet().getStringOptions().get("STREAK_COLOR_MODE");
        check(colorOption != null, "Flare color mode must be recognized by installed Iris");
        check(colorOption.getOption().getDefaultValue().equals("0"), "Current appearance must remain default color mode");
        check(colorOption.getOption().getAllowedValues().equals(List.of("0", "1")), "Color mode must allow exactly Current/Classic Blue 0/1");
        check(fixed.getOptionValues().getStringValueOrDefault("STREAK_COLOR_MODE").equals("0"), "Default mode must resolve to 0");
        // Menu tokens are option NAMES from the raw properties, not JCPP's
        // macro-expanded numeric values. ShaderProperties retains raw menu data.
        Properties menuProperties = new Properties();
        menuProperties.load(new StringReader(properties));
        String colorMenu = menuProperties.getProperty("screen.STREAK_SETTINGS", "");
        check(List.of(colorMenu.split("\\s+")).contains("STREAK_COLOR_MODE"), "Streak settings menu must expose color mode");
        var classic = options(shaders, roots, Map.of("STREAK_COLOR_MODE", "1"));
        check(classic.getOptionValues().getStringValueOrDefault("STREAK_COLOR_MODE").equals("1"), "Classic blue mode override must apply");
        checkImages(preprocess(properties, classic), true, "classic blue");
        check(List.of(colorMenu.split("\\s+")).contains("STREAK_COLOR_MODE"), "Classic blue still exposed in menu");
        System.out.println("PASS actual Iris color mode: default0, allowed0/1, explicit1, menu reference, both atlas mappings retained");

        var off = options(shaders, roots, Map.of("KINO_STREAK", "false"));
        check(!off.getOptionValues().getBooleanValueOrDefault("KINO_STREAK"), "OFF toggle must apply");
        checkImages(preprocess(properties, off), false, "fixed toggle OFF");
        System.out.println("PASS actual Iris toggle OFF: both images omitted");

        var zero = options(shaders, roots, Map.of("STREAK_INTENSITY", "0"));
        check(zero.getOptionValues().getBooleanValueOrDefault("KINO_STREAK"), "Intensity zero does not alter boolean");
        checkImages(preprocess(properties, zero), false, "fixed intensity zero");
        System.out.println("PASS actual Iris intensity zero: both images omitted");

        var on = options(shaders, roots, Map.of("KINO_STREAK", "true", "STREAK_INTENSITY", "100"));
        checkImages(preprocess(properties, on), true, "fixed explicit ON");
        System.out.println("PASS actual Iris explicit ON/intensity100: both images declared");
    }
}
