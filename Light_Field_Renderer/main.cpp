#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

#include "LF.h";
using namespace std;

int main() {
	LightField lf;
	lf.load("./toyLF/toyLF.conf");
	lf.test();

	system("pause");
	return 0;
}