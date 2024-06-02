#include "WindowHandler.h"

WindowHandler::WindowHandler(std::string windowName, int width, int height) : windowName(windowName), heigh(height), width(width) // Initialize member variable
{
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, width, height);
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::rectangle(image, cv::Point(width - 100, 0), cv::Point(width, 50), cv::Scalar(0, 0, 255), -1); // rectangle to exit the image
    cv::imshow(windowName, image);
    cv::setMouseCallback(windowName, WindowHandler::MouseEvent, this);
    cv::waitKey(0);
}

void WindowHandler::MouseEvent(int event, int x, int y, int, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN) {
        WindowHandler* self = static_cast<WindowHandler*>(userdata);
        if (x >= self->width - 100 && x <= self->width && y >= 0 && y <= 50) 
        {
            cv::destroyWindow(self->windowName);
        }
    }
}

WindowHandler::~WindowHandler()
{
    
}