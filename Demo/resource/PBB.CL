uchar lerp_uchar(uchar x, uchar y, float a) {
    return (uchar)((1.0f - a) * (float)x + a * (float)y);
}

__kernel void execute(__global uchar *original_image, // Obraz RGB
                           __global uchar *bloom_image, // 
                           __global uchar *final_image, // Wynikowy
                           const float bloomStrength,
                           const int width,
                           const int height) {

    int x = get_global_id(0); // Wspó³rzêdna X piksela
    int y = get_global_id(1); // Wspó³rzêdna Y piksela
    int index = (y * width + x) * 3; // Indeks w obrazie RGB (po 3 kana³y na piksel)
	if (x < width && y < height) {
		final_image[index] =lerp_uchar(original_image[index],bloom_image[index],bloomStrength);
		final_image[index+1] = lerp_uchar(original_image[index+1],bloom_image[index+1],bloomStrength);
		final_image[index+2] = lerp_uchar(original_image[index+2],bloom_image[index+2],bloomStrength);
	}

    
}
