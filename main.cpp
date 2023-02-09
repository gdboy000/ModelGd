#include "caffe_model.h"
#include "face_algorithm_demo1.h"
#include <ctime>
#include <iostream>
using namespace std;


int main()
{cout << "here" << "\n";
	cv::VideoCapture cp(0,cv::CAP_DSHOW);//易出错
	
	FaceIdentify *face = new FaceIdentify();
	while (1) {
		clock_t start = clock();
		cv::Mat frame;
		cp >> frame;
		cout << face->to_inference(frame) << "\n";

		clock_t end = clock();
		cout << "花费了" << (double)(end - start) / CLOCKS_PER_SEC << "秒" << endl;
	}
	delete face;
	return 0;
}