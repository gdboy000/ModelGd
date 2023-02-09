#include "caffe_model.h"

#ifdef _DEBUG
#include <iostream>
#endif // _DEBUG

using namespace std;
using namespace cv;

//������ִ�У�
//points��ʼ������ȡcaffeģ�ͣ�ѡ���Ƿ�cudn����
//Ӧ����ӽӿڣ�����ͷ��ţ��Ƿ�ѡ��cudn����

HandIdentify::HandIdentify( bool b = false) : use_cudn(b)
{
	points = vector<Point>(nPoints);//��Ϊ��Ҫ���������������
	this->net = dnn::readNetFromCaffe(this->protoFile, this->weightsFile);
	if (b) {//�Ƿ���Ҫcudn����
		this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}

}

HandIdentify::~HandIdentify()
{

}

//�ú�����ͨ���õ���frame����õ�ǰ֡���ֲ��������λ��
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
