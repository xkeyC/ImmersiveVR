// ImmersiveVR's GPU kernels, compiled ahead of time to kernels.ptx, which
// the binary embeds (gpu.rs):
//
//   nvcc -ptx -arch=compute_75 -o kernels.ptx kernels.cu
//
// resize_bgra: scales a BGRA picture (the capture to the eye size, and to the
// depth model's input).
//
// warp_nv12: both eyes from a picture and iw3 mlbw_l2 fields, straight to
// NV12: the warp of scripts/export_iw3_stereo.py (iw3's backward warp, BT.709
// limited range) in one pass. warp_rgba: the same, as two RGBA images.
//
// One thread per 2x2 block of one eye: four luma samples and the block's
// CbCr pair. Fields are upsampled on the fly (bilinear, align_corners) and
// each layer samples the picture along its row (bilinear, border clamp).

// One pixel (x, y) of `eye` warped: the eye's fields upsampled (bilinear,
// align corners) and two samples along the row (bilinear, border clamp),
// blended; BGR in 0..1.
__device__ float3 warp_pixel(
    const unsigned char* __restrict__ color, const float* __restrict__ fields,
    int width, int height, int fw, int fh, float shift, int eye, int x, int y)
{
    const size_t plane = (size_t)fh * fw;
    const float* f = fields + (size_t)eye * 4 * plane;
    const float sx = width > 1 ? (float)(fw - 1) / (float)(width - 1) : 0.0f;
    const float sy = height > 1 ? (float)(fh - 1) / (float)(height - 1) : 0.0f;
    const float fy = y * sy;
    const int y0 = (int)fy;
    const int y1 = min(y0 + 1, fh - 1);
    const float ay = fy - (float)y0;
    const float fx = x * sx;
    const int x0 = (int)fx;
    const int x1 = min(x0 + 1, fw - 1);
    const float ax = fx - (float)x0;
    float field[4];
    for (int c = 0; c < 4; c++) {
        const float* p = f + c * plane;
        const float top = p[y0 * fw + x0] * (1.0f - ax) + p[y0 * fw + x1] * ax;
        const float bottom = p[y1 * fw + x0] * (1.0f - ax) + p[y1 * fw + x1] * ax;
        field[c] = top * (1.0f - ay) + bottom * ay;
    }
    const unsigned char* row = color + (size_t)y * width * 4;
    float b = 0.0f, g = 0.0f, r = 0.0f;
    for (int layer = 0; layer < 2; layer++) {
        const float ix = fminf(fmaxf((float)x + field[layer] * shift, 0.0f), (float)(width - 1));
        const int i0 = (int)ix;
        const int i1 = min(i0 + 1, width - 1);
        const float t = ix - (float)i0;
        const unsigned char* a = row + i0 * 4;
        const unsigned char* c = row + i1 * 4;
        const float weight = field[2 + layer];
        b += fminf(fmaxf((a[0] * (1.0f - t) + c[0] * t) / 255.0f, 0.0f), 1.0f) * weight;
        g += fminf(fmaxf((a[1] * (1.0f - t) + c[1] * t) / 255.0f, 0.0f), 1.0f) * weight;
        r += fminf(fmaxf((a[2] * (1.0f - t) + c[2] * t) / 255.0f, 0.0f), 1.0f) * weight;
    }
    return make_float3(fminf(fmaxf(b, 0.0f), 1.0f), fminf(fmaxf(g, 0.0f), 1.0f), fminf(fmaxf(r, 0.0f), 1.0f));
}

extern "C" __global__ void warp_nv12(
    const unsigned char* __restrict__ color,  // BGRA, width x height, rows width * 4 bytes apart
    const float* __restrict__ fields,         // [8, fh, fw]: per eye offset0, offset1, weight0, weight1
    unsigned char* __restrict__ nv12,         // Y (width x 2 * height), then CbCr (width x height)
    int width, int height, int fw, int fh,
    float shift)                              // field offset -> pixels: delta_scale * (width - 1) / 2
{
    const int bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int by = blockIdx.y * blockDim.y + threadIdx.y;
    const int eye = blockIdx.z;
    if (bx * 2 >= width || by * 2 >= height) return;
    float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f;
    for (int dy = 0; dy < 2; dy++) {
        const int y = by * 2 + dy;
        for (int dx = 0; dx < 2; dx++) {
            const int x = bx * 2 + dx;
            const float3 bgr = warp_pixel(color, fields, width, height, fw, fh, shift, eye, x, y);
            const float luma = 16.0f + 219.0f * (0.2126f * bgr.z + 0.7152f * bgr.y + 0.0722f * bgr.x);
            nv12[((size_t)eye * height + y) * width + x] = (unsigned char)fminf(fmaxf(luma + 0.5f, 0.0f), 255.0f);
            sum_b += bgr.x;
            sum_g += bgr.y;
            sum_r += bgr.z;
        }
    }
    const float b = sum_b * 0.25f, g = sum_g * 0.25f, r = sum_r * 0.25f;
    const float cb = 128.0f + 224.0f * (-0.1146f * r - 0.3854f * g + 0.5f * b);
    const float cr = 128.0f + 224.0f * (0.5f * r - 0.4542f * g - 0.0458f * b);
    unsigned char* uv = nv12 + (size_t)width * 2 * height + ((size_t)eye * (height / 2) + by) * width + bx * 2;
    uv[0] = (unsigned char)fminf(fmaxf(cb + 0.5f, 0.0f), 255.0f);
    uv[1] = (unsigned char)fminf(fmaxf(cr + 0.5f, 0.0f), 255.0f);
}

// The same warp, each eye as an RGBA image (for an in-process consumer: the
// Unity plugin), gamma-encoded as the desktop is. One thread per pixel.
extern "C" __global__ void warp_rgba(
    const unsigned char* __restrict__ color,
    const float* __restrict__ fields,
    unsigned char* __restrict__ left,          // RGBA, width x height
    unsigned char* __restrict__ right,
    int width, int height, int fw, int fh,
    float shift)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int eye = blockIdx.z;
    if (x >= width || y >= height) return;
    const float3 bgr = warp_pixel(color, fields, width, height, fw, fh, shift, eye, x, y);
    unsigned char* out = (eye == 0 ? left : right) + ((size_t)y * width + x) * 4;
    out[0] = (unsigned char)(bgr.z * 255.0f + 0.5f);
    out[1] = (unsigned char)(bgr.y * 255.0f + 0.5f);
    out[2] = (unsigned char)(bgr.x * 255.0f + 0.5f);
    out[3] = 255;
}

// A tent filter as wide as the scale (bilinear convolution, as
// fast_image_resize's Bilinear): antialiased when shrinking, plain bilinear
// when enlarging. One thread per destination pixel; alpha is set opaque.
extern "C" __global__ void resize_bgra(
    const unsigned char* __restrict__ src, int sw, int sh,
    unsigned char* __restrict__ dst, int dw, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const float sx = (float)sw / (float)dw, sy = (float)sh / (float)dh;
    const float rx = fmaxf(sx, 1.0f), ry = fmaxf(sy, 1.0f);
    // Source pixel i's center is i + 0.5; the destination pixel's lies at
    // (x + 0.5) * scale, so the distance is i - c with:
    const float cx = (x + 0.5f) * sx - 0.5f, cy = (y + 0.5f) * sy - 0.5f;
    const int x0 = max((int)floorf(cx - rx) + 1, 0), x1 = min((int)ceilf(cx + rx) - 1, sw - 1);
    const int y0 = max((int)floorf(cy - ry) + 1, 0), y1 = min((int)ceilf(cy + ry) - 1, sh - 1);
    float b = 0.0f, g = 0.0f, r = 0.0f, total = 0.0f;
    for (int j = y0; j <= y1; j++) {
        const float wy = 1.0f - fabsf((float)j - cy) / ry;
        if (wy <= 0.0f) continue;
        const unsigned char* row = src + (size_t)j * sw * 4;
        for (int i = x0; i <= x1; i++) {
            const float w = (1.0f - fabsf((float)i - cx) / rx) * wy;
            if (w <= 0.0f) continue;
            const unsigned char* p = row + i * 4;
            b += p[0] * w;
            g += p[1] * w;
            r += p[2] * w;
            total += w;
        }
    }
    unsigned char* out = dst + ((size_t)y * dw + x) * 4;
    const float scale = total > 0.0f ? 1.0f / total : 0.0f;
    out[0] = (unsigned char)fminf(b * scale + 0.5f, 255.0f);
    out[1] = (unsigned char)fminf(g * scale + 0.5f, 255.0f);
    out[2] = (unsigned char)fminf(r * scale + 0.5f, 255.0f);
    out[3] = 255;
}

