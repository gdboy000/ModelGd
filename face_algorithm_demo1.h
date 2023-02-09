#pragma once
#ifndef FACE_ALGORITHM_DEMO1
#define FACE_ALGORITHM_DEMO1

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

using namespace cv;
class FaceIdentify
{
private:
	const char* model_path = "hand/res10_300x300_ssd_iter_140000.caffemodel";
	const char* prototxt_path = "hand/deploy.prototxt";
	dnn::Net net;
	int id;
	const cv::Ptr<face::LBPHFaceRecognizer>  recongnizer = face::LBPHFaceRecognizer::create();
	Mat image;
public:
	FaceIdentify();
	int to_inference(const Mat&);
};




#endif // !FACE_ALGORITHM_DEMO1

