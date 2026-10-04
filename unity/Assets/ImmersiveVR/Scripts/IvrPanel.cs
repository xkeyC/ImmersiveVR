// The control panel, ported from the WebXR client's: a world-space canvas a
// little below eye level in front of the viewer, pointed at with a controller
// ray and clicked with the trigger. Grip shows or hides it; while hidden, the
// trigger brings it back. Same layout and colours as the web panel, minus
// what PCVR has no use for (codec, passthrough, PC mute).

using System;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.TextCore.LowLevel;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.UI;

namespace ImmersiveVR
{
    public class IvrPanel : MonoBehaviour
    {
        [Tooltip("The screen this panel controls (default: the scene's)")]
        public IvrScreen screen;
        [Tooltip("Metres ahead of the viewer, and below eye level")]
        public float distance = 0.85f, drop = 0.28f;

        // Canvas pixels and their size in metres, as the web panel's.
        static readonly Vector2 Pixels = new Vector2(1280, 1040);
        const float Metres = 0.8f;

        static readonly Color Background = Rgb(18, 20, 26, 0.92f);
        static readonly Color Idle = Rgb(0x26, 0x2b, 0x35);
        static readonly Color Hover = Rgb(0x3a, 0x42, 0x52);
        static readonly Color Active = Rgb(0x4c, 0x8d, 0xff);
        static readonly Color TextColor = Rgb(0xe6, 0xe8, 0xec);
        static readonly Color Dim = Rgb(0x8a, 0x8f, 0x99);
        static readonly Color Dimmer = Rgb(0x6f, 0x76, 0x82);
        static readonly Color Warning = Rgb(0xff, 0xb3, 0x5c);

        static readonly int[] Resolutions = { 1080, 1440, 2160 };

        /// A setting changed in steps with − and +.
        class Stepper
        {
            public string label;
            public float step, min, max;
            public Func<IvrScreen, float> get;
            public Action<IvrScreen, float> set;
            public Func<float, string> format;
            public TMP_Text value;
        }

        static readonly Stepper[] Steppers =
        {
            new Stepper { label = "立体强度", step = 0.1f, min = 0, max = 10, get = s => s.divergence, set = (s, v) => s.divergence = v, format = v => $"{v:0.0} %" },
            new Stepper { label = "会聚（出屏 ↔ 入屏）", step = 0.05f, min = 0, max = 1, get = s => s.convergence, set = (s, v) => s.convergence = v, format = v => $"{v:0.00}" },
            new Stepper { label = "屏幕距离", step = 0.25f, min = 0.75f, max = 8, get = s => s.distance, set = (s, v) => s.distance = v, format = v => $"{v:0.00} m" },
            new Stepper { label = "屏幕宽度", step = 0.2f, min = 0.6f, max = 8, get = s => s.width, set = (s, v) => s.width = v, format = v => $"{v:0.0} m" },
            new Stepper { label = "曲率", step = 0.1f, min = 0, max = 1, get = s => s.curvature, set = (s, v) => s.curvature = v, format = v => v < 0.01f ? "平面" : $"{v:0.0}" },
            new Stepper { label = "高度", step = 0.1f, min = -1.5f, max = 1.5f, get = s => s.height, set = (s, v) => s.height = v, format = v => $"{(v >= 0 ? "+" : "")}{v:0.0} m" },
            new Stepper { label = "锐度", step = 0.1f, min = 0, max = 1, get = s => s.sharpness, set = (s, v) => s.sharpness = v, format = v => $"{v:0.0}" },
        };

        static TMP_FontAsset font;
        static Sprite rounded;

        Canvas canvas;
        TMP_Text status, details;
        readonly List<(int height, Button button)> resolutionButtons = new List<(int, Button)>();
        InputAction toggle, wake;
        bool visible = true;
        float renderFps;

        void Awake()
        {
            if (screen == null) screen = FindAnyObjectByType<IvrScreen>();
            Build();
        }

        void OnEnable()
        {
            if (screen != null) screen.Recentered += Place;
            // Either controller's grip toggles the panel; while hidden, the trigger shows it.
            toggle = new InputAction("IvrPanel toggle", InputActionType.Button);
            toggle.AddBinding("<XRController>{LeftHand}/{GripButton}");
            toggle.AddBinding("<XRController>{RightHand}/{GripButton}");
            toggle.performed += _ => Show(!visible);
            toggle.Enable();
            wake = new InputAction("IvrPanel wake", InputActionType.Button);
            wake.AddBinding("<XRController>{LeftHand}/{TriggerButton}");
            wake.AddBinding("<XRController>{RightHand}/{TriggerButton}");
            wake.performed += _ => { if (!visible) Show(true); };
            wake.Enable();
        }

        void OnDisable()
        {
            if (screen != null) screen.Recentered -= Place;
            toggle?.Dispose();
            wake?.Dispose();
        }

        void Show(bool show)
        {
            visible = show;
            canvas.gameObject.SetActive(show);
            if (show) Place();
        }

        /// In front of the viewer, a little below eye level, facing them; in the
        /// same space as the screen.
        void Place()
        {
            var viewer = screen != null && screen.Viewer != null ? screen.Viewer : Camera.main != null ? Camera.main.transform : null;
            if (viewer == null) return;
            if (screen != null && transform.parent != screen.transform.parent) transform.SetParent(screen.transform.parent, true);
            var forward = Vector3.ProjectOnPlane(viewer.forward, Vector3.up);
            if (forward.sqrMagnitude < 1e-4f) forward = Vector3.forward;
            forward.Normalize();
            transform.SetPositionAndRotation(
                viewer.position + forward * distance + Vector3.down * drop,
                Quaternion.LookRotation(forward, Vector3.up));
        }

        void Update()
        {
            renderFps = Mathf.Lerp(renderFps, 1f / Mathf.Max(Time.unscaledDeltaTime, 1e-4f), 0.05f);
            if (!visible || screen == null) return;
            var size = screen.PictureSize;
            SetText(status, screen.Problem ?? $"{size.x}×{size.y} · 新画面 {screen.PicturesPerSecond:0} fps",
                screen.Problem != null ? Warning : Dim);
            SetText(details, Details(), Dimmer);
            foreach (var stepper in Steppers) SetText(stepper.value, stepper.format(stepper.get(screen)), Color.white);
            foreach (var (height, button) in resolutionButtons) Paint(button, screen.resolution == height);
        }

        /// The runtime and how fast things run (where a stutter comes from).
        string Details()
        {
            var runtime = UnityEngine.XR.OpenXR.OpenXRRuntime.name;
            var parts = new List<string> { string.IsNullOrEmpty(runtime) ? "无 XR" : runtime };
            var displays = new List<UnityEngine.XR.XRDisplaySubsystem>();
            SubsystemManager.GetSubsystems(displays);
            if (displays.Count > 0 && displays[0].TryGetDisplayRefreshRate(out var hz)) parts.Add($"头显 {hz:0} Hz");
            parts.Add($"渲染 {renderFps:0} fps");
            if (UnityEngine.XR.XRSettings.enabled)
                parts.Add($"眼缓冲 {UnityEngine.XR.XRSettings.eyeTextureWidth}×{UnityEngine.XR.XRSettings.eyeTextureHeight}");
            return string.Join(" · ", parts);
        }

        static void SetText(TMP_Text text, string value, Color color)
        {
            if (text.text != value) text.text = value;
            if (text.color != color) text.color = color;
        }

        // ---- Building the canvas ------------------------------------------------

        void Build()
        {
            var root = new GameObject("Canvas", typeof(RectTransform));
            root.transform.SetParent(transform, false);
            canvas = root.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = Camera.main;
            root.AddComponent<CanvasScaler>().dynamicPixelsPerUnit = 2;
            // XR rays (and the mouse in the editor) hit its buttons.
            root.AddComponent<TrackedDeviceGraphicRaycaster>();
            var rect = (RectTransform)root.transform;
            rect.sizeDelta = Pixels;
            rect.localScale = Vector3.one * (Metres / Pixels.x);
            var background = root.AddComponent<Image>();
            Round(background, 36);
            background.color = Background;

            Label(root, "ImmersiveVR", 48, 56, 44, TextColor);
            status = Label(root, "", 48, 100, 26, Dim);
            details = Label(root, "", 48, 132, 22, Dimmer);

            float y = 150;
            const float row = 92;
            Label(root, "画面分辨率", 48, y + 34, 34, TextColor);
            for (var i = 0; i < Resolutions.Length; i++)
            {
                var height = Resolutions[i];
                var button = MakeButton(root, $"{height}p", 380 + i * 190, y, 170, 68, () =>
                {
                    screen.resolution = height;
                    screen.Save();
                });
                resolutionButtons.Add((height, button));
            }
            y += row;
            foreach (var stepper in Steppers)
            {
                var s = stepper;
                Label(root, s.label, 48, y + 34, 34, TextColor);
                MakeButton(root, "−", 380, y, 110, 68, () => Step(s, -1));
                s.value = Label(root, "", 640, y + 34, 34, Color.white, TextAlignmentOptions.Center, 240);
                MakeButton(root, "+", 790, y, 110, 68, () => Step(s, 1));
                y += row - 10;
            }
            y += 14;
            MakeButton(root, "重新居中", 48, y, 220, 72, () => screen.Recenter());
            MakeButton(root, "隐藏面板", 288, y, 220, 72, () => Show(false));
            MakeButton(root, "退出 VR", 528, y, 220, 72, Quit);
            Label(root, "握持键开关面板 · 面板隐藏时按扳机唤出", 48, Pixels.y - 36, 24, Dim);
        }

        void Step(Stepper stepper, int direction)
        {
            var value = stepper.get(screen) + direction * stepper.step;
            // On the step grid, so repeated steps do not drift.
            value = Mathf.Clamp(Mathf.Round(value / stepper.step) * stepper.step, stepper.min, stepper.max);
            stepper.set(screen, value);
            screen.Save();
        }

        static void Quit()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }

        /// A rect at canvas pixels (x, y from the top-left), `w x h`.
        static RectTransform At(GameObject parent, string name, float x, float y, float w, float h)
        {
            var child = new GameObject(name, typeof(RectTransform));
            child.transform.SetParent(parent.transform, false);
            var rect = (RectTransform)child.transform;
            rect.anchorMin = rect.anchorMax = rect.pivot = new Vector2(0, 1);
            rect.anchoredPosition = new Vector2(x, -y);
            rect.sizeDelta = new Vector2(w, h);
            return rect;
        }

        /// Text whose left edge (or centre) is at x and middle at y, as the
        /// web panel's canvas text.
        static TMP_Text Label(GameObject parent, string text, float x, float y, float size, Color color,
            TextAlignmentOptions alignment = TextAlignmentOptions.MidlineLeft, float width = 0)
        {
            var w = width > 0 ? width : Pixels.x - x - 32;
            var left = alignment == TextAlignmentOptions.Center ? x - w / 2 : x;
            var rect = At(parent, "Text", left, y - size, w, size * 2);
            var label = rect.gameObject.AddComponent<TextMeshProUGUI>();
            if (Font() != null) label.font = Font();
            label.text = text;
            label.fontSize = size;
            label.color = color;
            label.alignment = alignment;
            label.textWrappingMode = TextWrappingModes.NoWrap;
            label.overflowMode = TextOverflowModes.Ellipsis;
            label.raycastTarget = false;
            return label;
        }

        static Button MakeButton(GameObject parent, string text, float x, float y, float w, float h, Action click)
        {
            var rect = At(parent, "Button " + text, x, y, w, h);
            var image = rect.gameObject.AddComponent<Image>();
            Round(image, 14);
            var button = rect.gameObject.AddComponent<Button>();
            button.targetGraphic = image;
            Paint(button, false);
            button.onClick.AddListener(() => click());
            var label = Label(rect.gameObject, text, w / 2, h / 2, 32, Color.white, TextAlignmentOptions.Center, w);
            label.overflowMode = TextOverflowModes.Overflow;
            return button;
        }

        /// The web panel's button colours: idle, hovered, and active (chosen).
        static void Paint(Button button, bool active)
        {
            var colors = button.colors;
            var normal = active ? Active : Idle;
            if (colors.normalColor == normal && colors.fadeDuration == 0.05f) return;
            colors.normalColor = normal;
            colors.selectedColor = normal;
            colors.highlightedColor = active ? Active : Hover;
            colors.pressedColor = Active;
            colors.colorMultiplier = 1;
            colors.fadeDuration = 0.05f;
            button.colors = colors;
            ((Image)button.targetGraphic).color = Color.white;
        }

        static Color Rgb(int r, int g, int b, float a = 1) => new Color(r / 255f, g / 255f, b / 255f, a);

        // ---- Font and shapes ----------------------------------------------------

        /// A dynamic font from the system's CJK fonts (TextMesh Pro's default
        /// has no Chinese), made once.
        static TMP_FontAsset Font()
        {
            if (font != null) return font;
            var fonts = Environment.GetFolderPath(Environment.SpecialFolder.Fonts);
            foreach (var file in new[] { "msyh.ttc", "Deng.ttf", "simhei.ttf" })
            {
                var path = Path.Combine(fonts, file);
                if (!File.Exists(path)) continue;
                font = TMP_FontAsset.CreateFontAsset(path, 0, 90, 9, GlyphRenderMode.SDFAA, 2048, 2048);
                if (font == null) continue;
                font.isMultiAtlasTexturesEnabled = true;
                font.name = Path.GetFileNameWithoutExtension(file);
                return font;
            }
            Debug.LogWarning("[ImmersiveVR] no CJK system font found; the panel's Chinese will be missing");
            return null;
        }

        /// A rounded rectangle: a 9-sliced sprite made once, its corners scaled
        /// to `radius` canvas pixels.
        static void Round(Image image, float radius)
        {
            const int size = 64, corner = 24;
            if (rounded == null)
            {
                var texture = new Texture2D(size, size, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
                var pixels = new Color32[size * size];
                for (var y = 0; y < size; y++)
                for (var x = 0; x < size; x++)
                {
                    // Distance outside the rounded corner, anti-aliased over a pixel.
                    var dx = Mathf.Max(corner - x - 0.5f, x + 0.5f - (size - corner), 0);
                    var dy = Mathf.Max(corner - y - 0.5f, y + 0.5f - (size - corner), 0);
                    var alpha = Mathf.Clamp01(corner - Mathf.Sqrt(dx * dx + dy * dy) + 0.5f);
                    pixels[y * size + x] = new Color32(255, 255, 255, (byte)(alpha * 255));
                }
                texture.SetPixels32(pixels);
                texture.Apply();
                rounded = Sprite.Create(texture, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f), 100, 0,
                    SpriteMeshType.FullRect, new Vector4(corner, corner, corner, corner));
            }
            image.sprite = rounded;
            image.type = Image.Type.Sliced;
            image.pixelsPerUnitMultiplier = corner / radius;
        }
    }
}
