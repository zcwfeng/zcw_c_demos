


#ifndef Train_hpp
#define  Train_hpp

#include <string>

using namespace std;

#define ANN_ZH_XML "/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/HOG_ANN_ZH_DATA.xml"
#define ANN_XML "/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/HOG_ANN_DATA.xml"

#define ANN_CH_SAMPLE "/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/train/ann_zh"
#define ANN_SAMPLE "/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/resource/train/ann"

struct TrainStruct {
	string file;
	int label;
};
void doAnnZhTrain();
void doAnnTrain();

#endif /* Train_hpp */
