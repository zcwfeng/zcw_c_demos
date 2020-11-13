#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

using namespace cv;

class A {

};

class MyPtr {
public:
    MyPtr(A *a){}

    virtual ~MyPtr() {
        delete a;
    }

private:
    A *a;
};

int test() {
    Mat srcImage = imread("/Users/zcw/Downloads/WechatIMG12683.jpeg");
    if (!srcImage.data) {
        std::cout << "Image not loaded";
        return -1;
    }
    imshow("[img]", srcImage);
    waitKey(0);
    return 0;
}

void testCamera(){
    VideoCapture capture(0);
    while(1){
        Mat frame;
        capture >> frame;
        printf("Camera capture....\n");
        imshow("读取视频",frame);
        waitKey(30);
    }


}

int main() {
    std::cout << "Hello, World!" << std::endl;
    test();

    MyPtr ptr(new A());
//   testCamera();
    return 0;
}




