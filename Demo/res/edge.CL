__kernel void edges_detection(__global uchar *gray_image, // Obraz szaro-skalowy
                              __global uchar *final_image, // Obraz z krawêdziami
                              const int width,
                              const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Kernel matrices for Sobel operator
        const int horizontal_kernel_matrix[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
        const int vertical_kernel_matrix[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
        
        float gradientX = 0.0f;
        float gradientY = 0.0f;
        
        for(int i = -1; i <= 1; ++i) {
            for(int j = -1; j <= 1; ++j) {
                int pixel_value = gray_image[(y + i) * width + (x + j)];
                gradientX += pixel_value * horizontal_kernel_matrix[(i + 1) * 3 + (j + 1)];
                gradientY += pixel_value * vertical_kernel_matrix[(i + 1) * 3 + (j + 1)];
            }
        }
        
        float edge_strength = sqrt(gradientX * gradientX + gradientY * gradientY);
        final_image[y * width + x] = (uchar)min(255.0f, edge_strength); // Clamping the value
    } else {
        final_image[y * width + x] = 0;
    }
}