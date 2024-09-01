	float3 powVec3(float3 v, float p) {
        return (float3)(pow(v.x, p), pow(v.y, p), pow(v.z, p));
    }

    float3 toSRGB(float3 v) {
        const float invGamma = 1.0f / 2.2f;
        return powVec3(v, invGamma);
    }

    float sRGBToLuma(float3 col) {
        return dot(col, (float3)(0.299f, 0.587f, 0.114f));
    }

    float karisAverage(float3 col) {
        float luma = sRGBToLuma(toSRGB(col)) * 0.25f;
        return 1.0f / (1.0f + luma);
    }

	void checkEdge(int *x , int *y, int width, int height) {
		*x = max(0,*x);
		*x = min(width*2,*x);
		*y = max(0,*y);
		*y = min(height*2,*y);
	}
__kernel void execute(__global uchar *original_image, // Obraz RGB
                           __global uchar *final_image, // Wynikowy
                           const int width,
                           const int height) {

    int gid_x = get_global_id(0); // Wspó³rzêdna X piksela
    int gid_y = get_global_id(1); // Wspó³rzêdna Y piksela
    int index = (gid_y * width + gid_x) * 3; // Indeks w obrazie RGB (po 3 kana³y na piksel)
	
	
	
	int x=gid_x;
	int y=gid_y;
	int mgid_x = gid_x *2;
	int mgid_y = gid_y *2;
	if(mgid_x < width*2 && mgid_y < height*2) {
	x=((gid_x*2)-2);
	y=((gid_y*2)-2);
	checkEdge(&x,&y,width,height);
    int aID = (y*width*2 + x) * 3; 
	
	x=(gid_x*2);
    int bID = (y*width*2  + x) * 3; 
	
	x=(gid_x*2+2);
	checkEdge(&x,&y,width,height);
    int cID = (y*width*2  + x) * 3;
	
	x=(gid_x*2-2);
	y=(gid_y*2);
	checkEdge(&x,&y,width,height);	
    int dID = (y *width*2 + x) * 3; 
	
	x=(gid_x*2);
    int eID = (y*width*2  + x) * 3;
	
	x=(gid_x*2);
	checkEdge(&x,&y,width,height);	
    int fID = (y*width*2  + x) * 3; 
	
	x=(gid_x*2-2);
	y=(gid_y*2+2);
	checkEdge(&x,&y,width,height);
	int gID = (y*width*2  + x) * 3; 
	x=(gid_x*2);
    int hID = (y*width*2  + x) * 3;
	x=(gid_x*2+2);
	checkEdge(&x,&y,width,height);
    int iID = (y*width*2  + x) * 3;
	
	x=(gid_x*2 - 1);
	y=((gid_y*2-1));
	checkEdge(&x,&y,width,height);
    int jID = (y*width*2  + x) * 3;
	
	x=(gid_x*2+1);
	checkEdge(&x,&y,width,height);
    int kID = (y*width*2  + x) * 3;
	
	x=(gid_x*2-1);
	y=(gid_y*2+1);
	checkEdge(&x,&y,width,height);
    int lID = (y *width*2 + x) * 3;
	
	x=(gid_x*2+1);
	checkEdge(&x,&y,width,height);
    int mID = (y*width*2  + x) * 3; 
	
	
     if (gid_x < width && gid_y < height) {
		
		float3 a = (float3)(original_image[aID],original_image[aID+1],original_image[aID+2]);
		float3 b = (float3)(original_image[bID],original_image[bID+1],original_image[bID+2]);
		float3 c = (float3)(original_image[cID],original_image[cID+1],original_image[cID+2]);
		float3 d = (float3)(original_image[dID],original_image[dID+1],original_image[dID+2]);
		float3 e = (float3)(original_image[eID],original_image[eID+1],original_image[eID+2]);
		float3 f = (float3)(original_image[fID],original_image[fID+1],original_image[fID+2]);
		float3 g = (float3)(original_image[gID],original_image[gID+1],original_image[gID+2]);
		float3 h = (float3)(original_image[hID],original_image[hID+1],original_image[hID+2]);
		float3 i = (float3)(original_image[iID],original_image[iID+1],original_image[iID+2]);
		float3 j = (float3)(original_image[jID],original_image[jID+1],original_image[jID+2]);
		float3 k = (float3)(original_image[kID],original_image[kID+1],original_image[kID+2]);
		float3 l = (float3)(original_image[lID],original_image[lID+1],original_image[lID+2]);
		float3 m = (float3)(original_image[mID],original_image[mID+1],original_image[mID+2]);
		
		
		//float3 groups[5];
		
		//groups[0] = (a+b+d+e) * (0.125f/4.0f);
		//groups[1] = (b+c+e+f) * (0.125f/4.0f);
		//groups[2] = (d+e+g+h) * (0.125f/4.0f);
		//groups[3] = (e+f+h+i) * (0.125f/4.0f);
		//groups[4] = (j+k+l+m) * (0.5f/4.0f);
		//groups[0] *= karisAverage(groups[0]);
		//groups[1] *= karisAverage(groups[1]);
		//groups[2] *= karisAverage(groups[2]);
		//groups[3] *= karisAverage(groups[3]);
		//groups[4] *= karisAverage(groups[4]);
		 //downsample = 0.0f;//groups[0]+groups[1]+groups[2]+groups[3]+groups[4];
		//downsample = max(downsample, 0.01f);
		float3 downsample = e*0.125f;                // ok
		downsample += (a+c+g+i)*0.03125f;     // ok
		downsample += (b+d+f+h)*0.0625f;      // ok
		downsample += (j+k+l+m)*0.125f;   
		final_image[index] = downsample.r;
		final_image[index+1] = downsample.g;
		final_image[index+2] = downsample.b;
    }
   }

}
