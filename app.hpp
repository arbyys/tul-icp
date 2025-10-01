// icp.cpp 
// author: JJ

#pragma once
#include <opencv2/opencv.hpp>

class App {
public:
    App();

    bool init(void);
    int run(void);

    ~App();
private:
    cv::VideoCapture capture;
    void draw_cross_normalized(cv::Mat& img, cv::Point2f center_normalized, int size);
};

