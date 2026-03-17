#include <numeric>
#include <future>

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

float get_psnr(const cv::Mat& I1, const cv::Mat& I2)
{
    cv::Mat s1;
    cv::absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    cv::Scalar s = cv::sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

std::vector<uchar> lossy_quality_limit(const cv::Mat& frame, const float target_coefficient)
{
    std::mutex result_mutex;
    const unsigned max_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;
    std::vector<int> thread_results;
    std::vector<std::vector<uchar>> thread_compressed_frames;

    std::string suff(".jpg"); // target format
    if (!cv::haveImageWriter(suff))
        throw std::runtime_error("Can not compress to format:" + suff);


    // prepare parameters for JPEG compressor
    // we use only quality, but other parameters are available (progressive, optimization...)
    std::vector<int> compression_params_template;
    compression_params_template.push_back(cv::IMWRITE_JPEG_QUALITY);

    //std::cout << '[';

    //try step-by-step to decrease quality by 5%, until it fits into limit
    for (auto i = 5; i < 100; i += 5) {
        futures.push_back(std::async(std::launch::async, ([&, i, compression_params_template] {
            std::vector<int> compression_params;
            std::vector<uchar> bytes;
            compression_params = compression_params_template; // reset parameters
            compression_params.push_back(i);                  // set desired quality
            //std::cout << i << ',';

            // try to encode
            cv::imencode(suff, frame, bytes, compression_params);

            // check the size limit
            cv::Mat compressed_frame = cv::imdecode(bytes, cv::IMREAD_ANYCOLOR);
            float psnr = get_psnr(compressed_frame, frame);

            if (psnr >= target_coefficient) {
                std::lock_guard<std::mutex> lock(result_mutex);
                thread_results.push_back(i);
                thread_compressed_frames.push_back(bytes);
            }

        })));
    }


    for (auto& f : futures) f.get();

    // check if atleast one compression ratio worked
    if (thread_results.size() > 0)
    {
        // find iterator to smallest element
        auto min_it = std::min_element(thread_results.begin(), thread_results.end());

        // compute its index
        int index = std::distance(thread_results.begin(), min_it);

        return thread_compressed_frames[index];
    }
    // no acceptable compression found
    return {};
}

std::vector<uchar> lossy_bw_limit(const cv::Mat& frame, size_t size_limit)
{
    std::string suff(".jpg"); // target format
    if (!cv::haveImageWriter(suff))
        throw std::runtime_error("Can not compress to format:" + suff);

    std::vector<uchar> bytes;
    std::vector<int> compression_params;

    // prepare parameters for JPEG compressor
    // we use only quality, but other parameters are available (progressive, optimization...)
    std::vector<int> compression_params_template;
    compression_params_template.push_back(cv::IMWRITE_JPEG_QUALITY);

    //std::cout << '[';

    //try step-by-step to decrease quality by 5%, until it fits into limit
    for (auto i = 100; i > 0; i -= 5) {
        compression_params = compression_params_template; // reset parameters
        compression_params.push_back(i);                  // set desired quality
        //std::cout << i << ',';

        // try to encode
        cv::imencode(suff, frame, bytes, compression_params);

        // check the size limit
        if (bytes.size() <= size_limit)
            break; // ok, done 
    }
    //std::cout << "]\n";

    return bytes;
}