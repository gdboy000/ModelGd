
#include <iostream>
#include <ctime>
#include "face_algorithm_demo1.h"
/*
time:2022/4/15 19:15:21
author:14436
*/

using namespace std;

FaceIdentify::FaceIdentify()
{
    net = dnn::readNetFromCaffe(prototxt_path, model_path);
    recongnizer->read("hand/trainer.yml");
}

int FaceIdentify::to_inference(const Mat& frame)
{
    image = frame;
    Mat blob = dnn::blobFromImage(image, 1, Size(300, 300), Scalar(104, 117, 123));

    net.setInput(blob);
    Mat detections = net.forward();
    Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    for (int i = 0; i < detectionMat.rows; i++)
    {
        if (detectionMat.at<float>(i, 2) >= 0.5)
        {
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

            Rect object((int)xLeftBottom, (int)yLeftBottom,
                (int)(xRightTop - xLeftBottom),
                (int)(yRightTop - yLeftBottom));

            rectangle(image, object, Scalar(0, 255, 0));
            double n;
            int label = -1;
            recongnizer->predict(gray(object), label, n);
            if (n < 80)
            {
                return 7;
            }
        }
        else continue;
    }
    return -1;
}
