// Loads ivr_native.dll (the Rust 2D-to-3D pipeline) by hand instead of
// [DllImport]: the Unity editor never unloads a plugin it loaded itself, so
// every Rust rebuild would need an editor restart. In the editor a
// timestamped copy of the build output is loaded (cargo can keep writing the
// original) and unloaded when Play mode ends; the next Play loads the newest
// build. Players load the copy shipped next to them and never unload it.

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace ImmersiveVR
{
    public static class IvrNative
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct Info
        {
            public ulong frame;
            public ulong generation;
            public int width;
            public int height;
            public IntPtr left;
            public IntPtr right;
            public int running;
            public float divergence;
            public float convergence;
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int StringFn(byte[] utf8);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int PointerFn(IntPtr pointer);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int PollFn(ref Info info);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int LastErrorFn(byte[] buffer, int length);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate IntPtr RenderEventFuncFn();
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int ShutdownFn();

        [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
        static extern IntPtr LoadLibraryW(string path);
        [DllImport("kernel32", SetLastError = true)]
        static extern bool FreeLibrary(IntPtr module);
        [DllImport("kernel32", CharSet = CharSet.Ansi, SetLastError = true)]
        static extern IntPtr GetProcAddress(IntPtr module, string name);

        static IntPtr module;
        static string loadedCopy;
        static StringFn init, set;
        static PointerFn attach;
        static PollFn poll;
        static LastErrorFn lastError;
        static RenderEventFuncFn renderEventFunc;
        static ShutdownFn shutdown;

        public static bool Loaded => module != IntPtr.Zero;

#if UNITY_EDITOR
        // The module handle survives a domain reload only through SessionState.
        const string HandleKey = "ImmersiveVR.ivr_native.handle";
        const string CopyKey = "ImmersiveVR.ivr_native.copy";
#endif

        /// Loads the library at `path` (in the editor: a fresh copy of it).
        public static void Load(string path)
        {
            if (Loaded) return;
            if (!File.Exists(path)) throw new FileNotFoundException("ivr_native.dll not found (cargo build --release -p ivr-native)", path);
            var target = path;
#if UNITY_EDITOR
            Directory.CreateDirectory(Path.Combine(Application.temporaryCachePath, "ivr"));
            target = Path.Combine(Application.temporaryCachePath, "ivr", $"ivr_native_{DateTime.Now.Ticks}.dll");
            File.Copy(path, target, true);
#endif
            module = LoadLibraryW(target);
            if (module == IntPtr.Zero)
                throw new Exception($"LoadLibrary({target}) failed: error {Marshal.GetLastWin32Error()}");
            loadedCopy = target;
#if UNITY_EDITOR
            SessionState.SetString(HandleKey, module.ToInt64().ToString());
            SessionState.SetString(CopyKey, target);
#endif
            init = Bind<StringFn>("ivr_init");
            attach = Bind<PointerFn>("ivr_attach");
            set = Bind<StringFn>("ivr_set");
            poll = Bind<PollFn>("ivr_poll");
            lastError = Bind<LastErrorFn>("ivr_last_error");
            renderEventFunc = Bind<RenderEventFuncFn>("ivr_render_event_func");
            shutdown = Bind<ShutdownFn>("ivr_shutdown");
            Debug.Log($"[ImmersiveVR] loaded {target}");
        }

        static T Bind<T>(string name) where T : Delegate
        {
            var address = GetProcAddress(module, name);
            if (address == IntPtr.Zero) throw new Exception($"ivr_native has no {name}");
            return Marshal.GetDelegateForFunctionPointer<T>(address);
        }

        static byte[] Utf8(string text) => Encoding.UTF8.GetBytes((text ?? "") + "\0");

        public static string LastError()
        {
            if (!Loaded) return "not loaded";
            var buffer = new byte[2048];
            lastError(buffer, buffer.Length);
            var end = Array.IndexOf(buffer, (byte)0);
            return Encoding.UTF8.GetString(buffer, 0, end < 0 ? buffer.Length : end);
        }

        static void Check(int status, string what)
        {
            if (status != 0) throw new Exception($"{what}: {LastError()}");
        }

        /// Loads the models and starts capture and the pipeline (seconds: call off the main thread).
        public static void Init(string configJson) => Check(init(Utf8(configJson)), "ivr_init");

        /// The device of `texture` (any of Unity's own) receives the eye textures.
        public static void Attach(Texture texture) => Check(attach(texture.GetNativeTexturePtr()), "ivr_attach");

        public static void Set(string settingsJson) => Check(set(Utf8(settingsJson)), "ivr_set");

        public static Info Poll()
        {
            var info = new Info();
            Check(poll(ref info), "ivr_poll");
            return info;
        }

        public static IntPtr RenderEventFunc() => renderEventFunc();

        public static void Shutdown()
        {
            if (Loaded) Check(shutdown(), "ivr_shutdown");
        }

        /// Shuts the pipeline down and unloads the library (editor only; a
        /// player keeps it to the end). No render event may be queued.
        public static void Unload()
        {
#if UNITY_EDITOR
            if (!Loaded) return;
            try { Shutdown(); } catch (Exception e) { Debug.LogException(e); }
            FreeLibrary(module);
            module = IntPtr.Zero;
            init = set = null; attach = null; poll = null; lastError = null; renderEventFunc = null; shutdown = null;
            SessionState.EraseString(HandleKey);
            SessionState.EraseString(CopyKey);
            TryDelete(loadedCopy);
            Debug.Log("[ImmersiveVR] unloaded ivr_native");
#endif
        }

        static void TryDelete(string path)
        {
            try { if (!string.IsNullOrEmpty(path)) File.Delete(path); } catch (Exception) { }
        }

#if UNITY_EDITOR
        /// After a domain reload the statics are gone but the library may
        /// still be loaded: shut it down and unload it through the saved handle.
        [InitializeOnLoadMethod]
        static void RecoverAfterReload()
        {
            var saved = SessionState.GetString(HandleKey, "");
            if (saved == "" || Loaded || EditorApplication.isPlayingOrWillChangePlaymode) return;
            module = new IntPtr(long.Parse(saved));
            loadedCopy = SessionState.GetString(CopyKey, "");
            try
            {
                shutdown = Bind<ShutdownFn>("ivr_shutdown");
                lastError = Bind<LastErrorFn>("ivr_last_error");
            }
            catch (Exception) { }
            Unload();
        }

        [InitializeOnLoadMethod]
        static void UnloadAfterPlay()
        {
            EditorApplication.playModeStateChanged += state =>
            {
                // By EnteredEditMode the render thread has finished Play mode's frames.
                if (state == PlayModeStateChange.EnteredEditMode) Unload();
            };
        }
#endif
    }
}
