#pragma once
#include <string>
#include <opencv2/opencv.hpp>

class WindowHandler
{
public:
    WindowHandler(std::string windowName, int width, int height);
    static void MouseEvent(int event, int x, int y, int flags, void* userdata);
    ~WindowHandler();

private:
    std::string windowName;
    int width;
    int heigh;
};