//
//  SvmTrain.h
//  MyCarPlate
//
//  Created by lance on 2017/7/4.
//  Copyright © 2017年 lance. All rights reserved.
//

#ifndef SvmTrain_hpp
#define SvmTrain_hpp

#include <string>

using namespace std;


#define SVM_XML "/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/HOG_SVM_DATA.xml"



#define SVM_POS "/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/train/svm/posPlate/train"

#define SVM_NEG "/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/train/svm/negPlate/train"



struct TrainStruct{
    string file;
    int label;
};



void doSvmTrain();



#endif /* SvmTrain_hpp */
