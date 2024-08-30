#include "fileLoader/Toon.h"
#include "PBB.h"



int main() {

    toon_postprocess("swords.jpg","edges_detection_output.jpg");
    pbb_postprocess("swords.jpg", "PBB_output.jpg");
    return 0;
}