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
	void test2(int i = 0, int j = 0);
	void render();

	int height;
	int width;

	float camera_s;
	float camera_t;
	float disparity;
	float aperture; //sigma
private:
	void createGaussianKernel();
	void createGaussianKernel2();
	void renderByPixel(int i, int j);
	void renderByPixel(int i, int j, cv::Mat target);

	cv::Mat kernel;
	cv::Mat** data;
	cv::Mat result;
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
		this->data[i] = new cv::Mat[this->height];

	for (int i = 0; i < this->width; ++i)
	{
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
	for (int j = 0; j < this->height; ++j)
	{
		for (int i = 0; i < this->width; ++i)
		{
			cv::imshow("Test", this->data[i][j]);
			cv::waitKey(10);
			cout << "Show image i = " << i << "\t j = " << j << endl;
		}
		cv::waitKey(50);
	}
	cv::waitKey();
}

void LightField::test2(int i, int j)
{
	assert(this->data);
	cv::imshow("Test 2", this->data[i][j]);
}

void LightField::createGaussianKernel()
{
	float sigma = this->aperture;

	// odd ksize
	//int ksize = ((sigma - 0.8) / 0.3 + 1) * 2 + 1 + 0.5; // +0.5 for rounding
	//cv::Mat coefficient = cv::getGaussianKernel(ksize, sigma, CV_32F); // CV_64F by default

	// even ksize
	int ksize = ((this->aperture - 0.8) / 0.3 + 1) * 2; // +0.5 for rounding // TODO: 0.5 makes ksize odd
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
	this->kernel = temp * temp2;
}

void LightField::createGaussianKernel2()
{
	auto GaussianFunc = 
		[](float x0, float y0, float sigma, float x, float y) {
		return  exp(-(SQUARE(x - x0) + SQUARE(y - y0)) / (2.0f * SQUARE(sigma))) / (2 * CV_PI * SQUARE(sigma));
	};
	int ksize = (int)((this->aperture - 0.8) / 0.3 + 1) * 2;
	if (ksize == 0)
		ksize = 1;

	int s1 = this->camera_s;
	int t1 = this->camera_t;

	int s_lefttop = s1 - ksize / 2 + 1;
	int t_lefttop = t1 - ksize / 2 + 1;

	this->kernel = cv::Mat(ksize, ksize, CV_32F);
#pragma omp parallel for schedule(static)
	for (int i = 0; i < ksize; ++i)
		for (int j = 0; j < ksize; ++j)
			this->kernel.at<float>(j, i) = GaussianFunc(this->camera_s, this->camera_t, this->aperture, i + s_lefttop, j + t_lefttop);
	this->kernel /= cv::sum(this->kernel)[0];

}

void LightField::renderByPixel(int u, int v)
{
	this->renderByPixel(u, v, this->result);
}

void LightField::renderByPixel(int u, int v, cv::Mat target)
{
	int ksize = this->kernel.cols;
	if (ksize == 0)
	{
		this->createGaussianKernel2();
		ksize = this->kernel.cols;
	}

	int s1 = this->camera_s;
	int t1 = this->camera_t;

	int s_lefttop = s1 - ksize / 2 + 1;
	int t_lefttop = t1 - ksize / 2 + 1;

	float targetBGR[3];
	targetBGR[0] = targetBGR[1] = targetBGR[2] = 0.0f;

	for (int s = s_lefttop; s < s_lefttop + ksize; ++s)
	{
		for (int t = t_lefttop; t < t_lefttop + ksize; ++t)
		{
			int u_ = u - this->disparity * ((float)s - this->camera_s);
			int v_ = v - this->disparity * ((float)t - this->camera_t);

			float scale;
			uchar B, G, R;
			if (s < 0 || s >= this->width || t < 0 || t >= this->height)
			{
				scale = 0.0f;
				B = G = R = 0.0f;
			}
			else
			{
				scale = this->kernel.at<float>(t - t_lefttop, s - s_lefttop);
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
			
			targetBGR[0] += scale * B;
			targetBGR[1] += scale * G;
			targetBGR[2] += scale * R;
		}
	}

	uchar* pdst = target.ptr<uchar>(v);
	pdst[u * 3] = targetBGR[0];
	pdst[u * 3 + 1] = targetBGR[1];
	pdst[u * 3 + 2] = targetBGR[2];
}

void LightField::render()
{
	this->createGaussianKernel2();
	//this->result.setTo(cv::Scalar(0, 0, 0));
	cv::waitKey(1);
	cv::Mat buffer = cv::Mat(this->data[0][0].rows, this->data[0][0].cols, CV_8UC3, cv::Scalar(0));

#pragma omp parallel for schedule(guided)
	for (int i = 0; i < this->data[0][0].cols; ++i)
	{
		for (int j = 0; j < this->data[0][0].rows; ++j)
		{
			this->renderByPixel(i, j, buffer);
		}
	}
	cv::imshow("Result", buffer);
	this->result = buffer;
	//cv::waitKey(0);
}
