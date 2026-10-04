// The 3D desktop as a curved screen in front of the viewer: starts
// ivr_native (capture, depth, stereo on the GPU), shows its left and right
// eye textures through the StereoScreen shader, and passes the stereo
// settings to it.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.XR;

namespace ImmersiveVR
{
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public class IvrScreen : MonoBehaviour
    {
        [Header("Screen")]
        [Tooltip("Metres from the viewer to the screen's centre")]
        public float distance = 2.5f;
        [Tooltip("Screen width in metres (the height follows the picture)")]
        public float width = 2.4f;
        [Tooltip("0 = flat .. 1 = a cylinder around the viewer")]
        [Range(0f, 1f)] public float curvature = 0.3f;
        [Tooltip("Metres above eye level")]
        public float height = 0f;
        [Tooltip("The camera the screen is placed in front of (default: the main camera)")]
        public Transform viewer;

        [Header("3D")]
        [Range(0f, 10f)] public float divergence = 0.5f;
        [Range(0f, 1f)] public float convergence = 0.5f;
        [Tooltip("Eye picture height: 1080, 1440 or 2160 (never above the monitor's own)")]
        public int resolution = 2160;
        [Tooltip("0 = smooth .. 1 = sharp (the picture is usually shown smaller than it is; sharper picks finer mipmaps and may shimmer)")]
        [Range(0f, 1f)] public float sharpness = 0.5f;

        [Header("Source")]
        [Tooltip("A moving test pattern instead of the desktop")]
        public bool synthetic = false;
        [Tooltip("Monitor: 0 = primary, otherwise its Windows number")]
        public int monitor = 0;

        [Tooltip("ImmersiveVR/StereoScreen; referenced here so that builds include it")]
        public Shader screenShader;

        /// Placed again in front of the viewer (the panel follows).
        public event Action Recentered;
        /// Why there is no picture (starting, failed), or null.
        public string Problem { get; private set; } = "启动中：加载模型…";
        /// New pictures per second from the pipeline, and their size.
        public float PicturesPerSecond { get; private set; }
        public Vector2Int PictureSize { get; private set; }

        Material material;
        Mesh mesh;
        ulong countedFrame;
        float countedSince;
        Texture2D left, right, deviceProbe;
        ulong generation;
        Task starting;
        bool ready, failed;
        bool centered, present;
        float sentDivergence = -1, sentConvergence = -1;
        int sentResolution = -1;
        string meshKey = "";
        float aspect = 16f / 9f;

        /// The repository root in the editor (unity/ is inside it), the
        /// player's folder otherwise.
        static string Root =>
            Application.isEditor
                ? Path.GetFullPath(Path.Combine(Application.dataPath, "..", ".."))
                : Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

        static string LibraryPath =>
            Application.isEditor
                ? Path.Combine(Root, "target", "release", "ivr_native.dll")
                : Path.Combine(Application.dataPath, "Plugins", "x86_64", "ivr_native.dll");

        /// The settings the panel changes, kept between runs.
        const string PrefsPrefix = "ImmersiveVR.";

        void Load()
        {
            if (!PlayerPrefs.HasKey(PrefsPrefix + "divergence")) return;
            divergence = PlayerPrefs.GetFloat(PrefsPrefix + "divergence", divergence);
            convergence = PlayerPrefs.GetFloat(PrefsPrefix + "convergence", convergence);
            resolution = PlayerPrefs.GetInt(PrefsPrefix + "resolution", resolution);
            distance = PlayerPrefs.GetFloat(PrefsPrefix + "distance", distance);
            width = PlayerPrefs.GetFloat(PrefsPrefix + "width", width);
            curvature = PlayerPrefs.GetFloat(PrefsPrefix + "curvature", curvature);
            height = PlayerPrefs.GetFloat(PrefsPrefix + "height", height);
            sharpness = PlayerPrefs.GetFloat(PrefsPrefix + "sharpness", sharpness);
        }

        /// Keeps the current settings for the next run.
        public void Save()
        {
            PlayerPrefs.SetFloat(PrefsPrefix + "divergence", divergence);
            PlayerPrefs.SetFloat(PrefsPrefix + "convergence", convergence);
            PlayerPrefs.SetInt(PrefsPrefix + "resolution", resolution);
            PlayerPrefs.SetFloat(PrefsPrefix + "distance", distance);
            PlayerPrefs.SetFloat(PrefsPrefix + "width", width);
            PlayerPrefs.SetFloat(PrefsPrefix + "curvature", curvature);
            PlayerPrefs.SetFloat(PrefsPrefix + "height", height);
            PlayerPrefs.SetFloat(PrefsPrefix + "sharpness", sharpness);
            PlayerPrefs.Save();
        }

        void OnEnable()
        {
            Load();
            material = new Material(screenShader != null ? screenShader : Shader.Find("ImmersiveVR/StereoScreen"));
            GetComponent<MeshRenderer>().sharedMaterial = material;
            mesh = new Mesh { name = "IvrScreen" };
            GetComponent<MeshFilter>().sharedMesh = mesh;
            try
            {
                IvrNative.Load(LibraryPath);
                var config = Config();
                // Loading the models takes seconds: not on the main thread.
                starting = Task.Run(() => IvrNative.Init(config));
            }
            catch (Exception e)
            {
                Fail(e.Message);
                Debug.LogException(e);
            }
        }

        void Fail(string message)
        {
            failed = true;
            Problem = "出错：" + message;
        }

        /// ivr_init's JSON: models and runtime paths, capture and 3D settings.
        string Config()
        {
            var (ort, libs) = RuntimePaths();
            var parts = new List<string>
            {
                Field("models", Path.Combine(Root, "models", "depth")),
                Field("stereo_models", Path.Combine(Root, "models", "stereo")),
                Field("log_file", Path.Combine(Application.temporaryCachePath, "ivr_native.log")),
                $"\"monitor\": {monitor}",
                $"\"resolution\": {resolution}",
                $"\"divergence\": {divergence.ToString(CultureInfo.InvariantCulture)}",
                $"\"convergence\": {convergence.ToString(CultureInfo.InvariantCulture)}",
                $"\"synthetic\": {(synthetic ? "true" : "false")}",
                $"\"lib_dirs\": [{string.Join(", ", libs.ConvertAll(Quote))}]",
            };
            if (ort != null) parts.Add(Field("ort", ort));
            return "{" + string.Join(", ", parts) + "}";
        }

        static string Quote(string text) => "\"" + text.Replace("\\", "/").Replace("\"", "\\\"") + "\"";
        static string Field(string name, string value) => $"\"{name}\": {Quote(value)}";

        /// `ort=` and `lib=` lines of runtime/runtime.txt (this machine's ONNX
        /// Runtime and CUDA / TensorRT libraries), as the server reads them.
        static (string, List<string>) RuntimePaths()
        {
            string ort = null;
            var libs = new List<string>();
            var file = Path.Combine(Root, "runtime", "runtime.txt");
            if (File.Exists(file))
            {
                foreach (var raw in File.ReadAllLines(file))
                {
                    var line = raw.Trim();
                    if (line.StartsWith("#")) continue;
                    if (line.StartsWith("ort=")) ort = line.Substring(4).Trim();
                    else if (line.StartsWith("lib=")) libs.Add(line.Substring(4).Trim());
                }
            }
            else
            {
                var bundled = Path.Combine(Root, "runtime", "ort", "onnxruntime.dll");
                if (File.Exists(bundled)) ort = bundled;
            }
            return (ort, libs);
        }

        void Update()
        {
            if (failed) return;
            if (!ready)
            {
                if (starting == null || !starting.IsCompleted) return;
                if (starting.IsFaulted)
                {
                    Fail(starting.Exception?.GetBaseException().Message);
                    Debug.LogError($"[ImmersiveVR] {starting.Exception?.GetBaseException().Message}");
                    return;
                }
                deviceProbe = new Texture2D(4, 4, TextureFormat.RGBA32, false);
                deviceProbe.Apply();
                IvrNative.Attach(deviceProbe);
                ready = true;
                Problem = "等待画面…";
            }

            // Copies the newest pair into the eye textures on the render thread.
            GL.IssuePluginEvent(IvrNative.RenderEventFunc(), 0);
            var info = IvrNative.Poll();
            if (info.running == 0)
            {
                var error = IvrNative.LastError();
                Fail("管线停止：" + error);
                Debug.LogError($"[ImmersiveVR] pipeline stopped: {error}");
                return;
            }
            CountPictures(info);
            if (info.generation != generation && info.left != IntPtr.Zero && info.width > 0)
            {
                generation = info.generation;
                left = External(left, info.width, info.height, info.left);
                right = External(right, info.width, info.height, info.right);
                material.SetTexture("_LeftTex", left);
                material.SetTexture("_RightTex", right);
                aspect = (float)info.width / info.height;
            }
            FollowHeadset();
            SendSettings();
            material.SetFloat("_MipBias", -sharpness);
            UpdateMesh();
        }

        void CountPictures(IvrNative.Info info)
        {
            if (info.frame == 0) return;
            if (Problem != null) Problem = null;
            PictureSize = new Vector2Int(info.width, info.height);
            var elapsed = Time.unscaledTime - countedSince;
            if (countedFrame == 0) (countedFrame, countedSince) = (info.frame, Time.unscaledTime);
            else if (elapsed >= 1f)
            {
                PicturesPerSecond = (info.frame - countedFrame) / elapsed;
                (countedFrame, countedSince) = (info.frame, Time.unscaledTime);
            }
        }

        static Texture2D External(Texture2D texture, int width, int height, IntPtr view)
        {
            if (texture != null && texture.width == width && texture.height == height)
            {
                texture.UpdateExternalTexture(view);
                return texture;
            }
            if (texture != null) Destroy(texture);
            // A full mip chain (ivr_native generates it each frame); linear:
            // false: the views are sRGB.
            var created = Texture2D.CreateExternalTexture(width, height, TextureFormat.RGBA32, true, false, view);
            created.wrapMode = TextureWrapMode.Clamp;
            // Trilinear and anisotropic: the curved screen is seen at an angle
            // towards its sides.
            created.filterMode = FilterMode.Trilinear;
            created.anisoLevel = 16;
            return created;
        }

        void SendSettings()
        {
            if (divergence == sentDivergence && convergence == sentConvergence && resolution == sentResolution) return;
            (sentDivergence, sentConvergence, sentResolution) = (divergence, convergence, resolution);
            IvrNative.Set("{" +
                $"\"divergence\": {divergence.ToString(CultureInfo.InvariantCulture)}, " +
                $"\"convergence\": {convergence.ToString(CultureInfo.InvariantCulture)}, " +
                $"\"resolution\": {resolution}" + "}");
        }

        /// Recenters once the headset has a pose (the camera sits at the origin
        /// until then), and whenever the headset is put back on.
        void FollowHeadset()
        {
            var head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            if (!head.isValid)
            {
                // No XR (the editor without a headset): the camera as it is.
                if (!centered) { Recenter(); centered = true; }
                return;
            }
            if (!head.TryGetFeatureValue(CommonUsages.isTracked, out var tracked) || !tracked) return;
            var worn = !head.TryGetFeatureValue(CommonUsages.userPresence, out var presence) || presence;
            if (!centered || (worn && !present))
            {
                Recenter();
                centered = true;
            }
            present = worn;
        }

        /// Puts the screen in front of the viewer, level, facing them. It then
        /// belongs to the viewer's rig (the XR Origin): wherever the rig is
        /// moved, the screen keeps its place in the room.
        public void Recenter()
        {
            // Looked up now: in OnEnable the camera may not be enabled yet.
            if (viewer == null && Camera.main != null) viewer = Camera.main.transform;
            if (viewer == null) return;
            if (transform.parent != viewer.root && viewer.root != viewer) transform.SetParent(viewer.root, true);
            var forward = Vector3.ProjectOnPlane(viewer.forward, Vector3.up);
            if (forward.sqrMagnitude < 1e-4f) forward = Vector3.forward;
            transform.SetPositionAndRotation(viewer.position, Quaternion.LookRotation(forward.normalized, Vector3.up));
            Recentered?.Invoke();
        }

        /// The camera the screen was placed for.
        public Transform Viewer => viewer;

        /// A strip of columns along a cylinder arc (or flat), centred `distance`
        /// ahead; the picture's top row at the top (texture rows run top-down).
        void UpdateMesh()
        {
            var key = $"{distance}|{width}|{curvature}|{height}|{aspect}";
            if (key == meshKey) return;
            meshKey = key;
            const int columns = 64;
            var half = width / aspect / 2f;
            var vertices = new Vector3[(columns + 1) * 2];
            var uvs = new Vector2[vertices.Length];
            var triangles = new int[columns * 6];
            for (var i = 0; i <= columns; i++)
            {
                var u = (float)i / columns;
                float x, z;
                if (curvature < 0.01f)
                {
                    x = (u - 0.5f) * width;
                    z = distance;
                }
                else
                {
                    var radius = distance / curvature;
                    var angle = (u - 0.5f) * width / radius;
                    x = radius * Mathf.Sin(angle);
                    z = distance - radius + radius * Mathf.Cos(angle);
                }
                vertices[i * 2] = new Vector3(x, height + half, z);
                vertices[i * 2 + 1] = new Vector3(x, height - half, z);
                uvs[i * 2] = new Vector2(u, 0f);
                uvs[i * 2 + 1] = new Vector2(u, 1f);
                if (i < columns)
                {
                    var t = i * 6;
                    var v = i * 2;
                    triangles[t] = v; triangles[t + 1] = v + 2; triangles[t + 2] = v + 1;
                    triangles[t + 3] = v + 1; triangles[t + 4] = v + 2; triangles[t + 5] = v + 3;
                }
            }
            mesh.Clear();
            mesh.vertices = vertices;
            mesh.uv = uvs;
            mesh.triangles = triangles;
            mesh.RecalculateBounds();
        }

        void OnDisable()
        {
            // No more render events; the editor unloads the library once Play
            // mode has ended (IvrNative), a player keeps it running until quit.
            ready = false;
            if (!Application.isEditor && IvrNative.Loaded)
            {
                try { IvrNative.Shutdown(); } catch (Exception e) { Debug.LogException(e); }
            }
        }
    }
}
