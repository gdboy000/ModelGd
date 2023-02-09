#pragma once
#ifndef CAFFE_MODEL
#define CAFFE_MODEL

#include <dnn.hpp>
#include <imgproc.hpp>
#include <highgui.hpp>
#include <vector>


/*
作者：gaotonghe
注意：一定降低耦合性
*/
class HandIdentify {
    const int POSE_PAIRS[20][2] =
    {
        {0,1}, {1,2}, {2,3}, {3,4},         // thumb
        {0,5}, {5,6}, {6,7}, {7,8},         // index
        {0,9}, {9,10}, {10,11}, {11,12},    // middle
        {0,13}, {13,14}, {14,15}, {15,16},  // ring
        {0,17}, {17,18}, {18,19}, {19,20}   // small
    };
    const std::string protoFile = "hand/pose_deploy.prototxt";
    const std::string weightsFile = "hand/pose_iter_102000.caffemodel";
    const int nPoints = 22;
    const bool  use_cudn;
    int frameWidth;
    int frameHeight;
    float aspect_ratio;
    const int inHeight = 368;
    int inWidth;
    cv::dnn::Net net;
    cv::VideoCapture cap;
    cv::VideoWriter video;
    cv::Mat frame;

public:
    HandIdentify(bool);
    ~HandIdentify();
    void showVideoCopy(cv::Mat);
    std::vector<cv::Point> points;

};

#endif // !CAFFE_MODEL

