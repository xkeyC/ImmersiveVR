// The release build of the PCVR client: player settings (the web client's
// name, icon and colours), a Windows x64 player in target/unity/ImmersiveVR,
// ivr_native.dll copied in, and a zip in target/dist meant to be unpacked
// into the main release's folder (it shares runtime/ and models/ there).
// Menu: ImmersiveVR > Build Release. Build the DLL first:
//   cargo build --release -p ivr-native

using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace ImmersiveVR.EditorTools
{
    public static class ReleaseBuild
    {
        public const string Version = "0.0.1";
        const string Icon = "Assets/ImmersiveVR/Icons/icon.png";

        static string Repo => Path.GetFullPath(Path.Combine(Application.dataPath, "..", ".."));

        /// The web client's app info (manifest.webmanifest) for the player.
        [MenuItem("ImmersiveVR/Apply Player Settings")]
        public static void ApplyPlayerSettings()
        {
            PlayerSettings.productName = "ImmersiveVR";
            PlayerSettings.companyName = "ImmersiveVR";
            PlayerSettings.bundleVersion = Version;
            var icon = AssetDatabase.LoadAssetAtPath<Texture2D>(Icon);
            if (icon == null) throw new FileNotFoundException("the app icon is missing", Icon);
            PlayerSettings.SetIcons(NamedBuildTarget.Unknown, new[] { icon }, IconKind.Any);
            PlayerSettings.SplashScreen.backgroundColor = new Color32(0x0e, 0x0f, 0x12, 0xff);
            // A small window: the player captures the desktop it would otherwise
            // cover, and keeps running while other windows have the focus.
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.defaultScreenWidth = 1280;
            PlayerSettings.defaultScreenHeight = 720;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.runInBackground = true;
            PlayerSettings.visibleInBackground = true;
            PlayerSettings.forceSingleInstance = true;
            AssetDatabase.SaveAssets();
        }

        [MenuItem("ImmersiveVR/Build Release")]
        public static void Build()
        {
            var library = Path.Combine(Repo, "target", "release", "ivr_native.dll");
            if (!File.Exists(library))
                throw new FileNotFoundException("build it first: cargo build --release -p ivr-native", library);
            ApplyPlayerSettings();

            var folder = Path.Combine(Repo, "target", "unity", "ImmersiveVR");
            if (Directory.Exists(folder)) Directory.Delete(folder, true);
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray(),
                target = BuildTarget.StandaloneWindows64,
                locationPathName = Path.Combine(folder, "ImmersiveVR.exe"),
                options = BuildOptions.None,
            });
            if (report.summary.result != BuildResult.Succeeded)
                throw new Exception($"player build {report.summary.result}: {report.summary.totalErrors} errors");

            // Where IvrScreen loads it from in a player.
            var plugins = Path.Combine(folder, "ImmersiveVR_Data", "Plugins", "x86_64");
            Directory.CreateDirectory(plugins);
            File.Copy(library, Path.Combine(plugins, "ivr_native.dll"), true);
            Package(folder);
            Debug.Log($"[ImmersiveVR] release: {folder} ({report.summary.totalSize / 1048576} MB)");
        }

        /// Drops what must not ship (IL2CPP / Burst debug folders) and zips the
        /// player, its files at the zip's root.
        public static void Package(string folder = null)
        {
            folder ??= Path.Combine(Repo, "target", "unity", "ImmersiveVR");
            foreach (var debug in Directory.GetDirectories(folder)
                         .Where(d => d.EndsWith("_DoNotShip") || d.Contains("ButDontShipItWithYourGame")))
                Directory.Delete(debug, true);

            var dist = Path.Combine(Repo, "target", "dist");
            Directory.CreateDirectory(dist);
            var zip = Path.Combine(dist, $"ImmersiveVR-PCVR-v{Version}-windows-x64.zip");
            if (File.Exists(zip)) File.Delete(zip);
            ZipFile.CreateFromDirectory(folder, zip, System.IO.Compression.CompressionLevel.Optimal, false);
            Debug.Log($"[ImmersiveVR] packaged {zip} ({new FileInfo(zip).Length / 1048576} MB)");
        }
    }
}
