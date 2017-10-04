#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

using namespace std;

int main() {
	cout << "Hello World!" << endl;
	char* filename = "lena_std.tif";
	cv::Mat image = cv::imread(filename);
	cv::imshow(filename, image);
	cv::waitKey();
	system("pause");
	return 0;
}