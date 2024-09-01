__kernel void execute(__global uchar *image, // Obraz RGB
                            __global uchar *gray_image, // Obraz szaro-skalowy
                            const int width,
                            const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // Indeks w obrazie RGB
        uchar r = image[idx];
        uchar g = image[idx + 1];
        uchar b = image[idx + 2];
        
        uchar gray = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);

        // Zapisywanie wartoœci szaroœci w nowym buforze
        gray_image[y * width + x] = gray;
    }
}
