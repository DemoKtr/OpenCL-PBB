__kernel void execute(__global uchar *original_image, // Obraz RGB
                           __global uchar *edge_image,
                           __global uchar *final_image, // Obraz z krawêdziami
                           const int levels,
                           const int width,
                           const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int index = (y * width + x) * 3;

        // Odczyt krawêdzi
        int edge = (int)edge_image[index/3];
        
        if (edge < 245) {
            // Konwersja kolorów z uchar do float
            float R = (float)original_image[index] / 255.0f;
            float G = (float)original_image[index + 1] / 255.0f;
            float B = (float)original_image[index + 2] / 255.0f;

            // Kwantyzacja kolorów
            R = floor(R * levels) / levels;
            G = floor(G * levels) / levels;
            B = floor(B * levels) / levels;

            // Konwersja z powrotem do uchar
            final_image[index] = (uchar)clamp(R * 255.0f, 0.0f, 255.0f);
            final_image[index + 1] = (uchar)clamp(G * 255.0f, 0.0f, 255.0f);
            final_image[index + 2] = (uchar)clamp(B * 255.0f, 0.0f, 255.0f);
        } else {
            // Ustawienie koloru czarnego
            final_image[index] = 0;
            final_image[index + 1] = 0;
            final_image[index + 2] = 0;
        }
    }
    
   
}