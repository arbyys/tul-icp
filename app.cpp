// icp.cpp 
// Author: JJ

// C++ 
#include <iostream>
#include <chrono>
#include <stack>
#include <random>
#include <numeric>

// OpenCV 
#include <opencv2\opencv.hpp>

#include "app.hpp"
#include "utils.hpp"

void App::draw_cross_normalized(cv::Mat& img, cv::Point2f center_normalized, int size)
{
    center_normalized.x = std::clamp(center_normalized.x, 0.0f, 1.0f);
    center_normalized.y = std::clamp(center_normalized.y, 0.0f, 1.0f);
    size = std::clamp(size, 1, std::min(img.cols, img.rows));

    cv::Point2f center_absolute(center_normalized.x * img.cols, center_normalized.y * img.rows);

    cv::Point2f p1(center_absolute.x - size / 2, center_absolute.y);
    cv::Point2f p2(center_absolute.x + size / 2, center_absolute.y);
    cv::Point2f p3(center_absolute.x, center_absolute.y - size / 2);
    cv::Point2f p4(center_absolute.x, center_absolute.y + size / 2);

    cv::line(img, p1, p2, CV_RGB(255, 0, 0), 3);
    cv::line(img, p3, p4, CV_RGB(255, 0, 0), 3);
}

App::App()
{
    // default constructor
    // nothing to do here (so far...)
}

bool App::init(void)
{
    //open first available camera, using any API available (autodetect) 
    //capture = cv::VideoCapture(0, cv::CAP_ANY);

    //open video file
    capture = cv::VideoCapture("resources/video.mkv");

    if (!capture.isOpened())
    {
        std::cerr << "no source?" << std::endl;
        return false;
    }
    else
    {
        std::cout << "Source: " <<
            ": width=" << capture.get(cv::CAP_PROP_FRAME_WIDTH) <<
            ", height=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << '\n';
    }
    
    return true;
}

int App::run(void)
{
    cv::Mat frame, scene;

    while (1)
    {
        auto start = std::chrono::steady_clock::now();
        capture.read(frame);
        if (frame.empty())
        {
            std::cerr << "Cam disconnected? End of file?\n";
            break;
        }

        // show grabbed frame
        //cv::imshow("grabbed", frame);

        // WARNING: the original image MUST NOT be modified. If you want to draw into image,
        // do your own COPY!

        // analyze the image...
        // center = find_object(frame);

        // make a copy and draw center
        cv::Mat scene_cross;
        frame.copyTo(scene_cross);

        cv::Scalar threshold_lower = { 160, 130, 150 };
        cv::Scalar threshold_upper = { 180, 255, 255 };

        cv::Point2f center = find_object_chroma(scene_cross, threshold_lower, threshold_upper);
        draw_cross_normalized(scene_cross, center, 30);
        cv::imshow("scene", scene_cross);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() * 1000 << "ms" << std::endl;

        if (cv::waitKey(1) == 27)
            break;
    }

    return EXIT_SUCCESS;
}

App::~App()
{
    // clean-up
    cv::destroyAllWindows();
    std::cout << "Bye...\n";

    if (capture.isOpened())
        capture.release();
}
