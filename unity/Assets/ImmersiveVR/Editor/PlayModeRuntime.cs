// The OpenXR runtime Play mode uses, kept across editor restarts: Unity's own
// "Play Mode OpenXR Runtime" choice lives only in the editor process's
// environment and falls back to the system default on every start.
// Menu: ImmersiveVR > Play Mode Runtime.

using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace ImmersiveVR.EditorTools
{
    [InitializeOnLoad]
    static class PlayModeRuntime
    {
        const string PrefKey = "ImmersiveVR.PlayModeRuntime";
        const string SteamVR = @"C:\Program Files (x86)\Steam\steamapps\common\SteamVR\steamxr_win64.json";
        const string VirtualDesktop = @"C:\Program Files\Virtual Desktop Streamer\OpenXR\virtualdesktop-openxr.json";
        const string MetaLink = @"C:\Program Files\Oculus\Support\oculus-runtime\oculus_openxr_64.json";

        static PlayModeRuntime() => Apply(EditorPrefs.GetString(PrefKey, ""));

        /// The variables the OpenXR package's runtime selector sets: the
        /// loader reads XR_RUNTIME_JSON, the selector shows XR_SELECTED_RUNTIME_JSON.
        static void Apply(string json)
        {
            if (!string.IsNullOrEmpty(json) && !File.Exists(json))
            {
                Debug.LogWarning($"[ImmersiveVR] OpenXR runtime {json} is not installed; using the system default");
                json = "";
            }
            Environment.SetEnvironmentVariable("XR_RUNTIME_JSON", json);
            Environment.SetEnvironmentVariable("XR_SELECTED_RUNTIME_JSON", json);
        }

        static void Select(string json)
        {
            EditorPrefs.SetString(PrefKey, json);
            Apply(json);
            Debug.Log($"[ImmersiveVR] Play mode OpenXR runtime: {(json == "" ? "system default" : json)}");
        }

        [MenuItem("ImmersiveVR/Play Mode Runtime/SteamVR")]
        static void UseSteamVR() => Select(SteamVR);

        [MenuItem("ImmersiveVR/Play Mode Runtime/Virtual Desktop")]
        static void UseVirtualDesktop() => Select(VirtualDesktop);

        [MenuItem("ImmersiveVR/Play Mode Runtime/Meta Quest Link")]
        static void UseMetaLink() => Select(MetaLink);

        [MenuItem("ImmersiveVR/Play Mode Runtime/System Default")]
        static void UseSystemDefault() => Select("");

        [MenuItem("ImmersiveVR/Play Mode Runtime/SteamVR", true)]
        static bool CheckSteamVR() => Mark("ImmersiveVR/Play Mode Runtime/SteamVR", SteamVR);

        [MenuItem("ImmersiveVR/Play Mode Runtime/Virtual Desktop", true)]
        static bool CheckVirtualDesktop() => Mark("ImmersiveVR/Play Mode Runtime/Virtual Desktop", VirtualDesktop);

        [MenuItem("ImmersiveVR/Play Mode Runtime/Meta Quest Link", true)]
        static bool CheckMetaLink() => Mark("ImmersiveVR/Play Mode Runtime/Meta Quest Link", MetaLink);

        [MenuItem("ImmersiveVR/Play Mode Runtime/System Default", true)]
        static bool CheckSystemDefault() => Mark("ImmersiveVR/Play Mode Runtime/System Default", "");

        static bool Mark(string item, string json)
        {
            Menu.SetChecked(item, EditorPrefs.GetString(PrefKey, "") == json);
            return json == "" || File.Exists(json);
        }
    }
}
