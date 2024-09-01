#include "Toon.h"
#include "PBB.h"
#include "Mosaic.h"



int main() {

    toon_postprocess("swords.jpg","edges_detection_output.jpg");
    pbb_postprocess("swords.jpg", "PBB_output.jpg");
    mosaic_effect("swords.jpg", "Mosaic_output.jpg",5,30); //third argument should be x%10==0 and must be 
    return 0;
}