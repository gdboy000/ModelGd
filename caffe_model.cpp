#include "caffe_model.h"

#ifdef _DEBUG
#include <iostream>
#endif // _DEBUG

using namespace std;
using namespace cv;

//构造需执行：
//points初始化，读取caffe模型，选择是否cudn加速
//应当添加接口，摄像头序号，是否选择cudn加速

HandIdentify::HandIdentify( bool b = false) : use_cudn(b)
{
	points = vector<Point>(nPoints);//因为需要传出，不能申请堆
	this->net = dnn::readNetFromCaffe(this->protoFile, this->weightsFile);
	if (b) {//是否需要cudn加速
		this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}

}

HandIdentify::~HandIdentify()
{

}

//该函数可通过得到的frame来获得当前帧中手部点的所有位置
void HandIdentify::showVideoCopy(Mat frame)
{
	this->frameWidth = frame.cols;
	this->frameHeight = frame.rows;
	this->aspect_ratio = this->frameWidth / (float)this->frameHeight;
	this->inWidth = (int(this->aspect_ratio * inHeight));
	Mat inpBlob = dnn::blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
	this->net.setInput(inpBlob);
	Mat output = this->net.forward();
	int h = output.size[2];
	int w = output.size[3];

	for (int n = 0; n < nPoints; n++)
	{
		Mat probMap(h, w, CV_32F, output.ptr(0, n));
		resize(probMap, probMap, Size(this->frameWidth, this->frameHeight));
		Point max;
		double prob;
		minMaxLoc(probMap, 0, &prob, 0, &max);
		if (max.x <= 0 || max.y <= 0) continue;
		points[n] = max;
	}
}
