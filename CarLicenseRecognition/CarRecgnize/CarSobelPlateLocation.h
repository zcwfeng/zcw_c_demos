

#ifndef CarSobelPlateLocation_H
#define CarSobelPlateLocation_H
#include "CarPlateLocation.h"
class CarSobelPlateLocation: public CarPlateLocation {
public:
    CarSobelPlateLocation();
    ~CarSobelPlateLocation();

    // 1、要定位的图片 2、引用类型 作为定位结果
    void location(Mat src,vector<Mat>& dst);


};


#endif
