

	void checkEdge(int *x , int *y, int width, int height) {
		*x = max(0,*x);
		*x = min(width/2,*x);
		*y = max(0,*y);
		*y = min(height/2,*y);
	}
__kernel void execute(__global uchar *original_image, // Obraz RGB
                           __global uchar *final_image, // Wynikowy
                           const int width,
                           const int height) {

    int gid_x = get_global_id(0); // Wspó³rzêdna X piksela
    int gid_y = get_global_id(1); // Wspó³rzêdna Y piksela
    int index = (gid_y * width + gid_x) * 3; // Indeks w obrazie RGB (po 3 kana³y na piksel)
	

	if(gid_x < width && gid_y < height) {
	// Take 9 samples around current texel:
	// a - b - c
	// d - e - f
	// g - h - i
	// === ('e' is the current texel) ===

	int x = 0;
	int y = 0;
	x=(gid_x/2-1);
	y=(gid_y/2-1);
	checkEdge(&x,&y,width,height);
	int aID =(y * (width/2) + x) * 3;
	x=(gid_x/2);
	int bID =(y * (width/2) + x) * 3;
	
	x=(gid_x/2+1);
	checkEdge(&x,&y,width,height);
	int cID =(y * (width/2) + x) * 3;
	
	x=(gid_x/2)-1;
	y=gid_y/2;
	checkEdge(&x,&y,width,height);
	int dID =(y * (width/2) + x) * 3;
	x=gid_x/2;
	int eID =(y * (width/2) + x) * 3;
	x=(gid_x/2+1);
	checkEdge(&x,&y,width,height);
	int fID =(y * (width/2) + x) * 3;
	x=(gid_x/2-1);
	y=(gid_y/2+1);
	checkEdge(&x,&y,width,height);
	int gID =(y * (width/2) + x) * 3;
	x=(gid_x);
	checkEdge(&x,&y,width,height);
	int hID =(y * (width/2) + x) * 3;
	x=(gid_x/2+1);
	checkEdge(&x,&y,width,height);
	int iID =(y * (width/2) + x) * 3;
		
		float3 a = (float3)(original_image[aID],original_image[aID+1],original_image[aID+2]);
		float3 b = (float3)(original_image[bID],original_image[bID+1],original_image[bID+2]);
		float3 c = (float3)(original_image[cID],original_image[cID+1],original_image[cID+2]);
		float3 d = (float3)(original_image[dID],original_image[dID+1],original_image[dID+2]);
		float3 e = (float3)(original_image[eID],original_image[eID+1],original_image[eID+2]);
		float3 f = (float3)(original_image[fID],original_image[fID+1],original_image[fID+2]);
		float3 g = (float3)(original_image[gID],original_image[gID+1],original_image[gID+2]);
		float3 h = (float3)(original_image[hID],original_image[hID+1],original_image[hID+2]);
		float3 i = (float3)(original_image[iID],original_image[iID+1],original_image[iID+2]);
    
	
	float3 upsample = e*4.0f;
	upsample += (b+d+f+h)*2.0f;
	upsample += (a+c+g+i);
	upsample *= (1.0f / 16.0f);
	
	final_image[index] = upsample.r;
	final_image[index+1] = upsample.g;
	final_image[index+2] = upsample.b;
   }

}
