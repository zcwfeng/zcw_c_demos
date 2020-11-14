#ifndef CarColorPlateLocation_H
#define CarColorPlateLocation_H
#include "CarPlateLocation.h"
class CarColorPlateLocation :public CarPlateLocation {
public:
    CarColorPlateLocation();
    ~CarColorPlateLocation();

    // 1、要定位的图片 2、引用类型 作为定位结果
    void location(Mat src, vector<Mat>& dst);

};
#endif // CarColorPlateLocation_H

