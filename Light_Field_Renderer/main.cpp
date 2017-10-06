#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

#include "LF.h";
using namespace std;

void lightFieldApertureCallback(int value, LightField* lightfield);

int main() {
	LightField lf;
	lf.load("./toyLF/toyLF.conf");
	//lf.test();


	lf.camera_s = 8.5;
	lf.camera_t = 8.5;
	lf.disparity = 1.5;
	lf.aperture = 0.9;


	//lf.renderByPixel(50, 50);
	//cv::imshow("Result", lf.result);
	//cv::waitKey(0);

	cv::namedWindow("Result", CV_WINDOW_AUTOSIZE);
	int fp_init = 15;
	cv::createTrackbar("Focal Plane", "Result", &fp_init, 50, (cv::TrackbarCallback)lightFieldApertureCallback, &lf);
	lightFieldApertureCallback(fp_init, &lf);
	cv::waitKey();
	
	system("pause");
	return 0;
}

void lightFieldApertureCallback(int value, LightField* lightfield)
{
	lightfield->disparity = value / 10.0f + 0.1;
	cout << lightfield->disparity << endl;
	lightfield->render();
}