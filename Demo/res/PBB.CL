__kernel void toon_shading(__global uchar *original_image, // Obraz RGB
                           __global uchar *final_image, // Wynikowy
                           const int width,
                           const int height) {

    int gid_x = get_global_id(0); // Wspó³rzêdna X piksela
    int gid_y = get_global_id(1); // Wspó³rzêdna Y piksela
    int index = (gid_y * width + gid_x) * 3; // Indeks w obrazie RGB (po 3 kana³y na piksel)

    float2 srcTexelSize = (float2)(1.0f / width, 1.0f / height);
    float x = srcTexelSize.x;
    float y = srcTexelSize.y;

    // Przygotowanie funkcji pomocniczych
    float3 PowVec3(float3 v, float p) {
        return (float3)(pow(v.x, p), pow(v.y, p), pow(v.z, p));
    }

    float3 ToSRGB(float3 v) {
        const float invGamma = 1.0f / 2.2f;
        return PowVec3(v, invGamma);
    }

    float sRGBToLuma(float3 col) {
        return dot(col, (float3)(0.299f, 0.587f, 0.114f));
    }

    float KarisAverage(float3 col) {
        float luma = sRGBToLuma(ToSRGB(col)) * 0.25f;
        return 1.0f / (1.0f + luma);
    }

    // Pobranie pikseli z oryginalnego obrazu (sztucznie symuluj¹c funkcjê texture)
    #define getPixel(x, y) ((float3)(original_image[((y) * width + (x)) * 3 + 0], \
                                     original_image[((y) * width + (x)) * 3 + 1], \
                                     original_image[((y) * width + (x)) * 3 + 2]) / 255.0f)

    float3 a = getPixel(gid_x - 2*x, gid_y + 2*y);
    float3 b = getPixel(gid_x, gid_y + 2*y);
    float3 c = getPixel(gid_x + 2*x, gid_y + 2*y);

    float3 d = getPixel(gid_x - 2*x, gid_y);
    float3 e = getPixel(gid_x, gid_y);
    float3 f = getPixel(gid_x + 2*x, gid_y);

    float3 g = getPixel(gid_x - 2*x, gid_y - 2*y);
    float3 h = getPixel(gid_x, gid_y - 2*y);
    float3 i = getPixel(gid_x + 2*x, gid_y - 2*y);

    float3 j = getPixel(gid_x - x, gid_y + y);
    float3 k = getPixel(gid_x + x, gid_y + y);
    float3 l = getPixel(gid_x - x, gid_y - y);
    float3 m = getPixel(gid_x + x, gid_y - y);

    // Algorytm
    float3 downsample;
    int mipLevel = 0;  // Zmienna pomocnicza dla warunku

    float3 groups[5];
    switch (mipLevel) {
        case 0:
            groups[0] = (a + b + d + e) * (0.125f / 4.0f);
            groups[1] = (b + c + e + f) * (0.125f / 4.0f);
            groups[2] = (d + e + g + h) * (0.125f / 4.0f);
            groups[3] = (e + f + h + i) * (0.125f / 4.0f);
            groups[4] = (j + k + l + m) * (0.5f / 4.0f);

            for (int n = 0; n < 5; n++) {
                groups[n] *= KarisAverage(groups[n]);
            }

            downsample = groups[0] + groups[1] + groups[2] + groups[3] + groups[4];
            downsample = fmax(downsample, 0.0001f);
            break;
        default:
            downsample = e * 0.125f;
            downsample += (a + c + g + i) * 0.03125f;
            downsample += (b + d + f + h) * 0.0625f;
            downsample += (j + k + l + m) * 0.125f;
            break;
    }

    // Zapis wyniku
    final_image[index + 0] = (uchar)(downsample.x * 255.0f);
    final_image[index + 1] = (uchar)(downsample.y * 255.0f);
    final_image[index + 2] = (uchar)(downsample.z * 255.0f);
}
