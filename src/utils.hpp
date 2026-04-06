#pragma once
#include <opencv2/opencv.hpp>

cv::Point2f find_object_luma(cv::Mat& frame);

cv::Point2f find_object_chroma(
	cv::Mat& frame, 
	cv::Scalar& lower_threshold,
	cv::Scalar& upper_threshold
);

float get_psnr(const cv::Mat& I1, const cv::Mat& I2);

std::vector<uchar> lossy_quality_limit(const cv::Mat& frame, const float target_coefficient);

std::vector<uchar> lossy_bw_limit(const cv::Mat& frame, size_t size_limit);

void draw_cross_normalized(cv::Mat& img, cv::Point2f center_normalized, int size);
cv::Point2f find_face(cv::Mat& frame, cv::CascadeClassifier face_cascade);
std::vector<cv::Point2f> find_faces(cv::Mat& frame, cv::CascadeClassifier face_cascade);