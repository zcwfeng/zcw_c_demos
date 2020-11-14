#include <iostream>
#include "CarPlateRecgnize.h"

int main() {
    CarPlateRecgnize p("/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/HOG_SVM_DATA.xml",
                       "/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/HOG_ANN_ZH_DATA.xml",
                       "/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/HOG_ANN_DATA.xml");



    char path[100];
    for (int i = 1; i <8; ++i) {
         sprintf(path,"/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/test/test%d.jpg",i);
        Mat src = imread(path);
        cout << p.plateRecgnize(src) << endl;
        src.release();
    }
//    sprintf(path, "F:\\Lance\\OpenCV\\Car\\resource\\test\\test1.jpg");
//    //BGR格式的数据
//    Mat src = imread(path);
//    cout << p.plateRecgnize(src) << endl;


    return 0;
}