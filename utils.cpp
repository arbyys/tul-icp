#include <numeric>

#include <opencv2/opencv.hpp>

cv::Point2f find_object_luma(cv::Mat& frame) {
    // convert to grayscale, create threshold, sum white pixels
    // compute centroid of white pixels (average X,Y coordinate of all white pixels)
    cv::Point2f center;
    cv::Point2f center_normalized;
    unsigned int x_sum = 0;
    unsigned int y_sum = 0;
    unsigned int count = 0;

    for (int y = 0; y < frame.rows; y++) //y
    {
        for (int x = 0; x < frame.cols; x++) //x
        {
            // load source pixel
            cv::Vec3b pixel = frame.at<cv::Vec3b>(y, x);

            // compute temp grayscale value (convert from colors to Y)
            unsigned char Y = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];


            // FIND THRESHOLD (value 0..255)
            if (Y < 215) {
                // set output pixel black
                frame.at<cv::Vec3b>(y, x) = { 0, 0, 0 };
            }
            else {
                // set output pixel white
                frame.at<cv::Vec3b>(y, x) = { 255, 255, 255 };

                //update centroid...
                x_sum += x;
                y_sum += y;
                count += 1;
            }
        }
    }

    //std::cout << "x_sum: " << x_sum << std::endl;
    //std::cout << "y_sum: " << y_sum << std::endl;
    //std::cout << "count: " << count << std::endl;
    //std::cout << "image: " << frame.rows << "rows x " << frame.cols << "columns" << std::endl;
    float x_center = x_sum / count;
    float y_center = y_sum / count;

    //center = cv::Point2f(x_center, y_center);
    center_normalized = { x_center / frame.cols, y_center / frame.rows };

    return center_normalized;
}

cv::Point2f find_object_chroma(cv::Mat& frame, cv::Scalar& lower_threshold, cv::Scalar& upper_threshold) {
    cv::Mat scene_hsv;
    cv::cvtColor(frame, scene_hsv, cv::COLOR_BGR2HSV);

    cv::Mat scene_threshold;
    cv::inRange(scene_hsv, lower_threshold, upper_threshold, scene_threshold);

    //cv::namedWindow("scene_threshold", 0);
    //cv::imshow("scene_threshold", scene_threshold);

    std::vector<cv::Point> whitePixels;
    cv::findNonZero(scene_threshold, whitePixels);
    int whiteCnt = whitePixels.size();

    cv::Point whiteAccum = std::accumulate(whitePixels.begin(), whitePixels.end(), cv::Point(0.0, 0.0));

    cv::Point2f centroid_normalized(0.0f, 0.0f);
    if (whiteCnt > 0)
    {
        cv::Point centroid = { whiteAccum.x / whiteCnt, whiteAccum.y / whiteCnt };
        centroid_normalized = { static_cast<float>(centroid.x) / frame.cols, static_cast<float>(centroid.y) / frame.rows };
    }

    return centroid_normalized;
}
