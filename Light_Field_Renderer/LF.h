#pragma once
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

using namespace std;

class LightField
{
public:
	LightField();
	~LightField();
	void load(char* filename);
	void test();

	int height;
	int width;
	cv::Mat** data;
private:
	
};

LightField::LightField()
{

}

LightField::~LightField()
{
	for (int i = 0; i < this->width; ++i)
	{
		delete [] this->data[i];
	}
	delete [] this->data;
}

void LightField::load(char* filename)
{
	ifstream infile;
	infile.open(filename);
	infile >> this->width >> this->height;
	assert(this->width > 0 && this->height > 0);

	string buffer;
	infile.get(); // newline
	this->data = new cv::Mat*[this->width];
	for (int i = 0; i < this->width; ++i)
	{
		this->data[i] = new cv::Mat[this->height];
		for (int j = 0; j < this->height; ++j)
		{
			getline(infile, buffer);

			int first_sep = buffer.find('|');
			int last_sep = buffer.rfind('|');
			int x = stoi(buffer.substr(0, first_sep));
			int y = stoi(buffer.substr(first_sep + 1, last_sep - first_sep - 1));
			string path = buffer.substr(last_sep + 1);

			assert(x >= 0 && x < this->width);
			assert(y >= 0 && y < this->height);

			this->data[x][y] = cv::imread(path);
		}
	}
	
	infile.close();
}

void LightField::test()
{
	assert(this->data);
	for (int i = 0; i < this->width; ++i)
	{
		for (int j = 0; j < this->height; ++j)
		{
			cv::imshow("Test", this->data[i][j]);
			cv::waitKey(10);
			cout << "Show image i = " << i << "\t j = " << j << endl;
		}
		cv::waitKey(50);
	}
	cv::waitKey();
}