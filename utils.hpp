#pragma once
#include <opencv2/opencv.hpp>

cv::Point2f find_object_luma(cv::Mat& frame);

cv::Point2f find_object_chroma(
	cv::Mat& frame, 
	cv::Scalar& lower_threshold,
	cv::Scalar& upper_threshold
);
