#pragma once
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#define SQUARE(square_param) ((square_param)*(square_param))
#define RADIAN(radian_param) ((radian_param) * CV_PI / 180.0f)
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

	float fovy; //radians
	float rotate_xaxis;
	float rotate_yaxis;
	float rotate_zaxis;
	float translate_zaxis;
private:
	void createGaussianKernel();
	void createGaussianKernel2();
	void renderByPixel(int i, int j);
	void renderByPixel(int i, int j, cv::Mat target);
	void renderByPixel2(int u, int v, cv::Mat target);

	cv::Mat kernel;
	cv::Mat** raw_data;
	cv::Mat** warpped_data;
	cv::Mat result;

	static cv::Mat LightField::frustum(float left, float right, float bottom, float top, float near, float far);
	static cv::Mat LightField::perspective(float fovy, float aspect, float near, float far);
	static cv::Mat LightField::rotateX(float theta);
	static cv::Mat LightField::rotateY(float theta);
	static cv::Mat LightField::rotateZ(float theta);
	static cv::Mat LightField::translate(float deltaX, float deltaY, float deltaZ);
	static cv::Point2f LightField::performPerspective(cv::Point2f point, cv::Mat matrix);

	void LightField::updateCamera();
	cv::Mat original_border; //CV_32F
	cv::Mat current_border; //CV_32S
	float ratio;
	cv::Mat m_transform;
};

LightField::LightField()
{
	this->fovy = CV_PI / 2.0f;
	this->rotate_xaxis = 0.0f;
	this->rotate_yaxis = 0.0f;
	this->rotate_zaxis = 0.0f;
	this->translate_zaxis = 0.0f;
	this->current_border = cv::Mat(2, 4, CV_32S);
}

LightField::~LightField()
{
	for (int i = 0; i < this->width; ++i)
	{
		delete [] this->raw_data[i];
		delete[] this->warpped_data[i];
	}
	delete [] this->raw_data;
	delete [] this->warpped_data;
}

void LightField::load(char* filename)
{
	ifstream infile;
	infile.open(filename);
	infile >> this->width >> this->height;
	assert(this->width > 0 && this->height > 0);

	string buffer;
	infile.get(); // newline
	this->raw_data = new cv::Mat*[this->width];
	this->warpped_data = new cv::Mat*[this->width];

	for (int i = 0; i < this->width; ++i)
	{
		this->raw_data[i] = new cv::Mat[this->height];
		this->warpped_data[i] = new cv::Mat[this->height];
	}

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

			this->raw_data[x][y] = cv::imread(path);
			this->warpped_data[x][y] = this->raw_data[x][y].clone();
		}
	}
	
	infile.close();

	this->result = cv::Mat(this->raw_data[0][0].rows, this->raw_data[0][0].cols, CV_8UC3, cv::Scalar(0.0f, 0.0f, 0.0f));

	// Fancy Camera Initialization
	this->ratio = (float)this->raw_data[0][0].cols / (float)this->raw_data[0][0].rows; // width / height
	this->m_transform = cv::Mat::eye(3, 3, CV_32F);
	this->original_border = cv::Mat::ones(4, 4, CV_32F);

	this->original_border.at<float>(0, 0) = -this->raw_data[0][0].cols / 2; // -ratio;
	this->original_border.at<float>(1, 0) = this->raw_data[0][0].rows / 2; // 1.0f;

	this->original_border.at<float>(0, 1) = this->raw_data[0][0].cols / 2; // ratio;
	this->original_border.at<float>(1, 1) = this->raw_data[0][0].rows / 2; // 1.0f;

	this->original_border.at<float>(0, 3) = -this->raw_data[0][0].cols / 2; // ratio;
	this->original_border.at<float>(1, 3) = this->raw_data[0][0].rows / 2; // -1.0f;

	this->original_border.at<float>(0, 2) = this->raw_data[0][0].cols / 2;// ratio;
	this->original_border.at<float>(1, 2) = -this->raw_data[0][0].rows / 2; // -1.0f;

	this->updateCamera();
}

void LightField::test()
{
	assert(this->raw_data);
	for (int j = 0; j < this->height; ++j)
	{
		for (int i = 0; i < this->width; ++i)
		{
			cv::imshow("Test", this->raw_data[i][j]);
			cv::waitKey(10);
			cout << "Show image i = " << i << "\t j = " << j << endl;
		}
		cv::waitKey(50);
	}
	cv::waitKey();
}

void LightField::test2(int i, int j)
{
	assert(this->raw_data);
	cv::imshow("Test 2", this->raw_data[i][j]);
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
		float* praw_data = coefficient.ptr<float>(i);
		praw_data[0] = GaussianFunc(i, ksize, sigma);
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
					uchar* psrc = this->raw_data[s][t].ptr<uchar>(v_);
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

void LightField::renderByPixel2(int u, int v, cv::Mat target)
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

	//std::vector<cv::Point2f> src;
	//std::vector<cv::Point2f> dst;

	for (int s = s_lefttop; s < s_lefttop + ksize; ++s)
	{
		for (int t = t_lefttop; t < t_lefttop + ksize; ++t)
		{
			//src.push_back(cv::Point2f(s, t));
			//cv::perspectiveTransform(src, dst, this->m_transform);
			//s = dst[0].x;
			//t = dst[0].y;
			//dst.clear();
			int u_ = u - this->disparity * ((float)s - this->camera_s);
			int v_ = v - this->disparity * ((float)t - this->camera_t);

			//src.push_back(cv::Point2f(u_, v_));
			//cv::perspectiveTransform(src, dst, this->m_transform);
			//u_ = dst[1].x;
			//v_ = dst[1].y;

			//cv::Point2f new_uv = LightField::performPerspective(cv::Point2f(u_, v_), this->m_transform);
			//u_ = (int)new_uv.x;
			//v_ = (int)new_uv.y;


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
					uchar* psrc = this->warpped_data[s][t].ptr<uchar>(v_);
					B = psrc[u_ * 3];
					G = psrc[u_ * 3 + 1];
					R = psrc[u_ * 3 + 2];
				}
			}

			targetBGR[0] += scale * B;
			targetBGR[1] += scale * G;
			targetBGR[2] += scale * R;

			//s = src[0].x;
			//t = src[0].y;
			//src.clear();
			//dst.clear();
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
	this->updateCamera();
	//this->result.setTo(cv::Scalar(0, 0, 0));
	cv::waitKey(1);
	cv::Mat buffer = cv::Mat(this->raw_data[0][0].rows, this->raw_data[0][0].cols, CV_8UC3, cv::Scalar(0));

#pragma omp parallel for schedule(guided)
	for (int i = 0; i < this->raw_data[0][0].cols; ++i)
	{
		for (int j = 0; j < this->raw_data[0][0].rows; ++j)
		{
			this->renderByPixel2(i, j, buffer);
		}
	}

	// TODO REMOVE..

	cv::polylines(buffer, this->current_border.t(), true, cv::Scalar(0, 0, 255));
	// TODO END
	cv::imshow("Result", buffer);
	this->result = buffer;
	//cv::waitKey(0);
}

cv::Mat LightField::frustum(float left, float right, float bottom, float top, float near, float far)
{
	cv::Mat ret = cv::Mat::zeros(4, 4, CV_32F);
	ret.at<float>(0, 0) = 2 * near / (right - left);

	ret.at<float>(1, 1) = 2 * near / (top - bottom);

	ret.at<float>(0, 2) = (right + left) / (right - left);
	ret.at<float>(1, 2) = (top + bottom) / (top - bottom);
	ret.at<float>(2, 2) = -(far + near) / (far - near);
	ret.at<float>(3, 2) = -1.0f;

	ret.at<float>(2, 3) = -2 * far * near / (far - near);

	return ret;
}

cv::Mat LightField::perspective(float fovy, float aspect, float near, float far) // fovy in radians   projective * view * model * col
{
	float top = tan(fovy / 2.0f) * near;
	float bottom = -top;
	float right = top * aspect;
	float left = -top * aspect;
	return LightField::frustum(left, right, bottom, top, near, far);
}

cv::Mat LightField::rotateX(float theta) //A*x
{
	cv::Mat ret = cv::Mat::eye(4, 4, CV_32F);
	ret.at<float>(1, 1) = cos(theta);
	ret.at<float>(2, 1) = -sin(theta);
	ret.at<float>(1, 2) = sin(theta);
	ret.at<float>(2, 2) = cos(theta);
	return ret;
}

cv::Mat LightField::rotateY(float theta)
{
	cv::Mat ret = cv::Mat::eye(4, 4, CV_32F);
	ret.at<float>(0, 0) = cos(theta);
	ret.at<float>(0, 2) = -sin(theta);
	ret.at<float>(2, 0) = sin(theta);
	ret.at<float>(2, 2) = cos(theta);
	return ret;
}

cv::Mat LightField::rotateZ(float theta)
{
	cv::Mat ret = cv::Mat::eye(4, 4, CV_32F);
	ret.at<float>(0, 0) = cos(theta);
	ret.at<float>(1, 0) = -sin(theta);
	ret.at<float>(0, 1) = sin(theta);
	ret.at<float>(1, 1) = cos(theta);
	return ret;
}

cv::Mat LightField::translate(float deltaX, float deltaY, float deltaZ)
{
	cv::Mat ret = cv::Mat::eye(4, 4, CV_32F);
	ret.at<float>(0, 3) = deltaX;
	ret.at<float>(1, 3) = deltaY;
	ret.at<float>(2, 3) = deltaZ;
	return ret;
}

void LightField::updateCamera()
{
	cv::Mat m_projective = LightField::perspective(fovy, 1.0f, 1.0f, 100.0f); //this->ratio
	cv::Mat m_view = LightField::translate(0.0f, 0.0f, this->translate_zaxis) * LightField::rotateX(this->rotate_xaxis) * LightField::rotateY(this->rotate_yaxis) * LightField::rotateZ(this->rotate_zaxis);
	//TODO
	cv::Mat border; // 4 * cols
	border = m_projective * m_view * this->original_border;
	
	cv::Point2f srcQuad[4];
	cv::Point2f dstQuad[4];

	for (int i = 0; i < 4; i++)
	{
		srcQuad[i] = cv::Point2f(this->original_border.at<float>(0, i), this->original_border.at<float>(1, i));
		float x = border.at<float>(0, i) / -border.at<float>(3, i);
		float y = border.at<float>(1, i) / -border.at<float>(3, i);
		dstQuad[i] = cv::Point2f(x, y);
		this->current_border.at<int>(0, i) = x + this->raw_data[0][0].cols / 2;
		this->current_border.at<int>(1, i) = y + this->raw_data[0][0].rows / 2;

	}

	vector<cv::Point2f> pts_src;
	vector<cv::Point2f> pts_dst;
	for (int i = 0; i < 4; ++i)
	{
		pts_src.push_back(cv::Point2f((float)srcQuad[i].x, (float)srcQuad[i].y));
		pts_dst.push_back(cv::Point2f((float)dstQuad[i].x, (float)dstQuad[i].y));
	}

	this->m_transform = cv::getPerspectiveTransform(srcQuad, dstQuad); // CV_64F
	this->m_transform = cv::findHomography(pts_src, pts_dst);

	for (int i = 0; i < this->width; ++i)
	{
		for (int j = 0; j < this->height; ++j)
		{
			cv::warpPerspective(this->raw_data[i][j], this->warpped_data[i][j], this->m_transform, cv::Size(this->raw_data[0][0].cols, this->raw_data[0][0].rows), cv::InterpolationFlags::INTER_NEAREST | cv::InterpolationFlags::WARP_INVERSE_MAP);
		}
	}
	//cout << m_projective << endl;
	//cout << this->original_border << endl;
	//cout << border << endl;
	cout << this->m_transform << endl;
}

cv::Point2f LightField::performPerspective(cv::Point2f point, cv::Mat matrix)
{
	assert(matrix.cols == 3 && matrix.rows == 3);
	cv::Mat point_ = cv::Mat::ones(3, 1, CV_64F); // matrix is CV_64F
	point_.at<double>(0, 0) = point.x;
	point_.at<double>(1, 0) = point.y;

	cv::Mat result = matrix * point_;
	double w = result.at<double>(2, 0);
	cv::Point2f ret = cv::Point2f(result.at<double>(0, 0) / w, result.at<double>(1, 0) / w);
	return ret;
}
