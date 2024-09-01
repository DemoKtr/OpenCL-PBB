inline uchar uchar_clamp(int value, int min_val, int max_val) {
    // Konwertuj wartoœci min i max do int, aby unikn¹æ problemów z przekroczeniem zakresu
    int clamped_value = value < min_val ? min_val : (value > max_val ? max_val : value);
    
    // Upewnij siê, ¿e wynik mieœci siê w zakresie uchar
    if (clamped_value < 0) {
        return 0;
    } else if (clamped_value > 255) {
        return 255;
    } else {
        return (uchar)clamped_value;
    }
}

__kernel void execute(__global uchar *original_image, // Obraz RGB
                           __global uchar *final_image, // Wynikowy
                           __global int *random, // Wynikowy
                           const int step,
                           const int width,
                           const int height,
                           const int originalwidth,
                           const int originalheight) {

    int gid_x = get_global_id(0); // Wspó³rzêdna X piksela
    int gid_y = get_global_id(1); // Wspó³rzêdna Y piksela

	

	if(gid_x < width && gid_y < height) {
		float R = 0.0f;
		float G = 0.0f;
		float B = 0.0f;
		int total_weight = 0;
		for(int i = -(step/2); i <= (step/2); ++i) {
			for(int j = -(step/2); j <= (step/2); ++j) {
				int weight =  abs((abs(i + 1) + abs(j + 1))-(abs(step/2)*2)/2);  
				total_weight += weight;
				int x = gid_x * step + j;
				int y = gid_y * step + i;
				if (x >= 0 && x < originalwidth && y >= 0 && y < originalheight) {				
					R += (float)(original_image[((y * (width*step) + x) * 3)] * weight) / 255.0f;
					G += (float)(original_image[((y * (width*step) + x) * 3)+1 ] * weight) / 255.0f;
					B += (float)(original_image[((y * (width*step) + x) * 3)+2 ] * weight) / 255.0f;
				}
			}
		}
		
		
		
		R /=total_weight;
		G /=total_weight;
		B /=total_weight;
	
		int random_index = gid_y * width + gid_x;  // Przyk³adowy sposób ustalania indeksu tablicy random
		R += (float)(random[random_index]/255.0f);
		G += (float)(random[random_index]/255.0f);
		B += (float)(random[random_index]/255.0f);
		uchar finalColorR = (uchar)clamp(R * 255.0f, 0.0f, 255.0f);
		uchar finalColorG = (uchar)clamp(G * 255.0f, 0.0f, 255.0f);
		uchar finalColorB = (uchar)clamp(B * 255.0f, 0.0f, 255.0f);
		

		
		for(int i = -(step/2); i <= (step/2); ++i) {
			for(int j = -(step/2); j <= (step/2); ++j) {
				int x = gid_x * step + j;
				int y = gid_y * step + i;
				if (x >= 0 && x < originalwidth && y >= 0 && y < originalheight) {
					final_image[((y * (width*step) + x) * 3)] = finalColorR;
					final_image[((y * (width*step) + x) * 3) + 1] = finalColorG;
					final_image[((y * (width*step) + x) * 3) + 2] = finalColorB;
				}
			}
		}
   }

}




