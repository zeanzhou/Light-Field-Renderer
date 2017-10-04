#pragma once
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>

#define SQUARE(square_param) ((square_param)*(square_param))
using namespace std;

class LightField
{
public:
	LightField();
	~LightField();
	void load(char* filename);
	void test();
	void renderByPixel(int i, int j);

	int height;
	int width;
	cv::Mat** data;
	cv::Mat result;

	float camera_s;
	float camera_t;
	float disparity;
	float aperture; //sigma
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

	this->result = cv::Mat(this->data[0][0].rows, this->data[0][0].cols, CV_8UC3, cv::Scalar(0.0f, 0.0f, 0.0f));
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

void LightField::renderByPixel(int u, int v)
{
	float sigma = this->aperture;
	
	// odd ksize
	//int ksize = ((sigma - 0.8) / 0.3 + 1) * 2 + 1 + 0.5; // +0.5 for rounding
	//cv::Mat coefficient = cv::getGaussianKernel(ksize, sigma, CV_32F); // CV_64F by default

	// even ksize
	int ksize = ((sigma - 0.8) / 0.3 + 1) * 2 + 0.5; // +0.5 for rounding
	auto GaussianFunc = [](int index, int ksize, float sigma) { return  exp(-(SQUARE(index - (ksize - 1) / 2.0f)) / (2.0f * SQUARE(sigma))); };
	cv::Mat coefficient = cv::Mat(ksize, 1, CV_32F);
	for (int i = 0; i < ksize; ++i) 
	{
		float* pdata = coefficient.ptr<float>(i);
		pdata[0] = GaussianFunc(i, ksize, sigma);
	}
	coefficient /= cv::sum(coefficient)[0];

	cv::Mat temp = coefficient * cv::Mat::ones(1, ksize, CV_32F); // [a;b;c;d] * [1 1 1 1]
	cv::Mat temp2 = cv::Mat::diag(coefficient); // [a;b;c;d] on the diag
	cv::Mat kernel = temp * temp2;
	
	int s1 = this->camera_s;
	int t1 = this->camera_t;

	int s_lefttop = s1 - ksize / 2 + 1;
	int t_lefttop = t1 - ksize / 2;

	uchar* pdst = this->result.ptr<uchar>(v);

	for (int s = s_lefttop; s < s_lefttop + ksize; ++s)
	{
		for (int t = t_lefttop; t < t_lefttop + ksize; ++t)
		{
			int u_ = u + this->disparity * (s - this->camera_s);
			int v_ = v + this->disparity * (t - this->camera_t);
			
			// check stuv
			float scale;
			uchar B, G, R;
			if (s < 0 || s >= this->width || t < 0 || t >= this->height)
			{
				scale = 0.0f;
				B = G = R = 0.0f;
			}
			else
			{
				scale = kernel.at<float>(t, s);
				if (u_ < 0 || u_ >= this->result.cols || v_ < 0 || v_ >= this->result.rows)
					B = G = R = 0.0f;
				else
				{
					uchar* psrc = this->data[s][t].ptr<uchar>(v_);
					B = psrc[u_ * 3];
					G = psrc[u_ * 3 + 1];
					R = psrc[u_ * 3 + 2];
				}
			}
			
			pdst[u * 3] += scale * B;
			pdst[u * 3 + 1] += scale * G;
			pdst[u * 3 + 2] += scale * R;
		}
	}

}