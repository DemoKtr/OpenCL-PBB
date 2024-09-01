__kernel void execute(__global uchar *original_image, // Obraz RGB
                           __global uchar *final_image, // Wynikowy
                           const int step,
                           const int width,
                           const int height) {

    int gid_x = get_global_id(0); // Wspó³rzêdna X piksela
    int gid_y = get_global_id(1); // Wspó³rzêdna Y piksela
    int channel = get_global_id(2); // Wspó³rzêdna Y piksela
    int index = (((gid_y*step) * width + (gid_x*step)) * 3)+channel; // Indeks w obrazie RGB (po 3 kana³y na piksel)
	

	if(gid_x < width && gid_y < height && gid_x > 0 && gid_y > 0) {
		float result = 0.0f;
		float total_weight = 0.0f;
		for(int i=-(step/2);i<(step/2);++i){
			for(int j=-(step/2);j<(step/2);++j){
				float weight = 1.0f / (abs(i + 1) * abs(j + 1));  
				total_weight += weight;  
				result += (float)(original_image[((((gid_y * step) + j) * width + ((gid_x * step) + i)) * 3) + channel] * weight) / 255.0f;
			}
		}
		result /= total_weight;
		uchar finalColor = (uchar)clamp(result * 255.0f, 0.0f, 255.0f);
		
		for(int i=-(step/2);i<(step/2);++i){
			for(int j=-(step/2);j<(step/2);++j){
				original_image[((((gid_y * step) + j) * width + ((gid_x * step) + i)) * 3) + channel] = finalColor;
			}
		}
   }

}




