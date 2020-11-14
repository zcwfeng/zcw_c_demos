#ifndef  CarPlateLocation_H
#define CarPlateLocation_H
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class CarPlateLocation {
public:
	CarPlateLocation();
	virtual ~CarPlateLocation();
protected:
	int verifySizes(RotatedRect rotatedRect);
	void tortuosity(Mat src, vector<RotatedRect> &rects, vector<Mat> &dst_plates);
	void safeRect(Mat src, RotatedRect rect, Rect2f  &dst_rect);
	void rotation(Mat src, Mat &dst, Size rect_size, Point2f center, double angle);
};
#endif // ! CarPlateLocation_H

