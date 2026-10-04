// The UI language: Chinese when the system's is (zh, zh_CN, zh-TW, …),
// English otherwise, as the web client decides by the browser's.

using System.Globalization;
using UnityEngine;

namespace ImmersiveVR
{
    public static class Lang
    {
        public static readonly bool Zh = IsChinese();

        /// The text in the UI language.
        public static string T(string chinese, string english) => Zh ? chinese : english;

        static bool IsChinese()
        {
            switch (Application.systemLanguage)
            {
                case SystemLanguage.Chinese:
                case SystemLanguage.ChineseSimplified:
                case SystemLanguage.ChineseTraditional:
                    return true;
            }
            var name = CultureInfo.CurrentUICulture.Name.Replace('-', '_').ToLowerInvariant();
            return name == "zh" || name.Contains("zh_");
        }
    }
}
