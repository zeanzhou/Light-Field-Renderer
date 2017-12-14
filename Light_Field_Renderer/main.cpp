#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

#include "LF.h";
using namespace std;

void lightFieldFocalPlaneCallback(int value, LightField* lightfield);
void lightFieldSCallback(int value, LightField* lightfield);
void lightFieldTCallback(int value, LightField* lightfield);
void lightFieldRxCallback(int value, LightField* lightfield);
void lightFieldRyCallback(int value, LightField* lightfield);
void lightFieldTzCallback(int value, LightField* lightfield);
void lightFieldApertureCallback(int value, LightField* lightfield);

//void lightFieldSTCallback(int value, LightField* lightfield);
//int global_s, global_t;

int main() {
	LightField lf;
	lf.load("./toyLF/toyLF.conf");
	//lf.test();

	lf.camera_s = 8.5;
	lf.camera_t = 8.5;
	lf.disparity = 1.5;
	lf.aperture = 0.7;

	cv::namedWindow("Result", CV_WINDOW_AUTOSIZE);
	int fp_init = 15;
	int cs_init = lf.camera_s * 10;
	int ct_init = lf.camera_t * 10;
	int a_init = lf.aperture * 10 - 7;
	int rx_init = 90;
	int ry_init = 90;
	int tz_init = 50;

	cv::createTrackbar("Focal Plane", "Result", &fp_init, 1500, (cv::TrackbarCallback)lightFieldFocalPlaneCallback, &lf);
	lightFieldFocalPlaneCallback(fp_init, &lf);

	cv::createTrackbar("Camera S", "Result", &cs_init, (lf.width - 1) * 10, (cv::TrackbarCallback)lightFieldSCallback, &lf);
	cv::createTrackbar("Camera T", "Result", &ct_init, (lf.height - 1) * 10, (cv::TrackbarCallback)lightFieldTCallback, &lf);
	lightFieldSCallback(cs_init, &lf);
	lightFieldTCallback(ct_init, &lf);

	cv::createTrackbar("Aperture", "Result", &a_init, 23, (cv::TrackbarCallback)lightFieldApertureCallback, &lf);
	lightFieldApertureCallback(a_init, &lf);

	cv::createTrackbar("X axis Rotation", "Result", &rx_init, 180, (cv::TrackbarCallback)lightFieldRxCallback, &lf);
	lightFieldRxCallback(rx_init, &lf);

	cv::createTrackbar("Y axis Rotation", "Result", &ry_init, 180, (cv::TrackbarCallback)lightFieldRyCallback, &lf);
	lightFieldRyCallback(ry_init, &lf);

	cv::createTrackbar("Z axis Translation", "Result", &tz_init, 100, (cv::TrackbarCallback)lightFieldTzCallback, &lf);
	lightFieldTzCallback(tz_init, &lf);

	cv::waitKey();

	// test st
	//cv::namedWindow("Test 2", CV_WINDOW_AUTOSIZE);
	//cv::createTrackbar("s-value", "Test 2", &global_s, lf.width - 1, (cv::TrackbarCallback)lightFieldSTCallback, &lf);
	//cv::createTrackbar("t-value", "Test 2", &global_t, lf.height - 1, (cv::TrackbarCallback)lightFieldSTCallback, &lf);
	//lightFieldSTCallback(0, &lf);
	//cv::waitKey();
	system("pause");
	return 0;
}

void lightFieldFocalPlaneCallback(int value, LightField* lightfield)
{
	lightfield->disparity = value / 100.0f;
	cout << "New Focal Plane (disparity) = " << lightfield->disparity << endl;
	lightfield->render();
}

void lightFieldSCallback(int value, LightField* lightfield)
{
	lightfield->camera_s = value / 10.0f;
	cout << "New s = " << lightfield->camera_s << endl;
	lightfield->render();
}

void lightFieldTCallback(int value, LightField* lightfield)
{
	lightfield->camera_t = value / 10.0f;
	cout << "New t = " << lightfield->camera_t << endl;
	lightfield->render();
}

void lightFieldRxCallback(int value, LightField* lightfield)
{
	lightfield->rotate_xaxis = RADIAN(value - 90.0f);
	cout << "Rotate around x-axis = " << value - 90.0f << endl;
	lightfield->render();
}

void lightFieldRyCallback(int value, LightField* lightfield)
{
	lightfield->rotate_yaxis = RADIAN(value - 90.0f);
	cout << "Rotate around y-axis = " << value - 90.0f << endl;
	lightfield->render();
}

void lightFieldTzCallback(int value, LightField* lightfield)
{
	lightfield->translate_zaxis = (value - 50) / 10.0f;
	cout << "Translate along z-axis = " << (value - 50) / 10.0f << endl;
	lightfield->render();
}


void lightFieldApertureCallback(int value, LightField* lightfield)
{
	lightfield->aperture = value / 10.0f + 0.7;
	cout << "New aperture (sigma) = " << lightfield->aperture << endl;
	lightfield->render();
}

//void lightFieldSTCallback(int value, LightField* lightfield)
//{
//	lightfield->test2(global_s, global_t);
//}
