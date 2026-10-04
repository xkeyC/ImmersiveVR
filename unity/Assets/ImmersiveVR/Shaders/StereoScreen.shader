// The 3D desktop: each eye samples its own picture (ivr_native's left and
// right eye textures). Works with single-pass instanced and multi-pass
// stereo rendering; unlit, as the picture is already final.
Shader "ImmersiveVR/StereoScreen"
{
    Properties
    {
        _LeftTex ("Left eye", 2D) = "black" {}
        _RightTex ("Right eye", 2D) = "black" {}
        _MipBias ("Mip bias (negative: sharper)", Range(-2, 1)) = -0.5
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue" = "Geometry" }
        Pass
        {
            Name "Unlit"
            Cull Off
            ZWrite On

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_LeftTex);
            SAMPLER(sampler_LeftTex);
            TEXTURE2D(_RightTex);
            SAMPLER(sampler_RightTex);
            float _MipBias;

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                ZERO_INITIALIZE(Varyings, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                output.uv = input.uv;
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                // The textures are sRGB: sampled to linear, written to the
                // eye buffers as the pipeline expects in Linear color space.
                half3 color = unity_StereoEyeIndex == 0
                    ? SAMPLE_TEXTURE2D_BIAS(_LeftTex, sampler_LeftTex, input.uv, _MipBias).rgb
                    : SAMPLE_TEXTURE2D_BIAS(_RightTex, sampler_RightTex, input.uv, _MipBias).rgb;
                return half4(color, 1);
            }
            ENDHLSL
        }
    }
}
