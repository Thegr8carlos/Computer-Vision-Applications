#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void onMouse(int event, int x, int y, int, void* userdata)
{
    if (event != cv::EVENT_LBUTTONDOWN) {
        return;
    }

    cv::Mat* image = reinterpret_cast<cv::Mat*>(userdata);

    cv::Vec3b rgb_value = image->at<cv::Vec3b>(y, x);
    int r = rgb_value[2];
    int g = rgb_value[1];
    int b = rgb_value[0];


    cv::Mat rgb_pixel(1, 1, CV_8UC3, rgb_value);
    cv::Mat hsv_pixel;
    cv::cvtColor(rgb_pixel, hsv_pixel, cv::COLOR_BGR2HSV);
    cv::Vec3b hsv_value = hsv_pixel.at<cv::Vec3b>(0, 0);
    int h = hsv_value[0];
    int s = hsv_value[1];
    int v = hsv_value[2];

 
    std::cout << "RGB: (" << r << ", " << g << ", " << b << ") ";
    std::cout << "HSV: (" << h << ", " << s << ", " << v << ")" << std::endl;
}


cv::Mat createHSVChart(int width, int height) 
{
    cv::Mat hsv_chart(height, width, CV_8UC3);

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            uchar hue = static_cast<uchar>(180.0 * i / width);
            uchar saturation = static_cast<uchar>(255.0 * j / height);
            uchar value = 255;
            hsv_chart.at<cv::Vec3b>(j, i) = cv::Vec3b(hue, saturation, value);
        }
    }

    cv::cvtColor(hsv_chart, hsv_chart, cv::COLOR_HSV2BGR);
    return hsv_chart;
}


void segmentFrame(cv::Mat& frame, const cv::Scalar& lower_bound, const cv::Scalar& upper_bound, cv::Mat background) {
    cv::Mat hsv_frame, mask; // stores the hsv frame and the mask filter
    cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_frame, lower_bound, upper_bound, mask); // checks all the values in the range and sets to 0 the values that not match
    std::vector<cv::Point> points; // used to store the points that are non zero
    cv::findNonZero(mask, points); // storing the points in the vector
    for (auto p : points)
    {
        cv::Vec3b rgb_value = background.at<cv::Vec3b>(p.y, p.x); // gets the background color of the background 
        frame.at<cv::Vec3b>(p.y, p.x) = rgb_value; // sets the value of the backgrounf in the frame 
    }
}



int main() {

    // Creates and shows the hsv graphic
    cv::Mat hsv_chart = createHSVChart(800, 600);
    cv::namedWindow("HSV Chart", cv::WINDOW_AUTOSIZE);
    cv::imshow("HSV Chart", hsv_chart);
    
    // reads the background image
    cv::Mat background = cv::imread("back.jpg");



    // opens the video 
    cv::VideoCapture cap("test1.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error al abrir el video" << std::endl;
        return -1;
    }

    //  windows
    cv::namedWindow("Original Video", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("HSV Video", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Final video", cv::WINDOW_AUTOSIZE);

    cv::Mat frame; // each frame of the video
    cv::Mat hsv_frame; // each hsv frame of the video 
    
    // range to apply the change.
    cv::Scalar lower_bound(20, 20, 0); 
    cv::Scalar upper_bound(90, 250, 255);
    
    
    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat segmented_frame = frame.clone(); // copy of frame  


        segmentFrame(segmented_frame, lower_bound, upper_bound, background); // function to apply the transformation of the frame

        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV); // parsing to hsv to show the transformation in all the frame

        // shows each frame
        cv::imshow("Original Video", frame);
        cv::imshow("HSV Video", hsv_frame);
        cv::imshow("Final video", segmented_frame);

        // mouse events
        cv::setMouseCallback("Original Video", onMouse, &frame);

        if (cv::waitKey(30) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
