// icp.cpp 
// author: JJ

#pragma once
#include <vector>
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
    cv::CascadeClassifier face_cascade = cv::CascadeClassifier("resources/haarcascade_frontalface_default.xml");
    cv::Point2f find_face(cv::Mat& frame);
    std::vector<cv::Point2f> find_faces(cv::Mat& frame);
};

