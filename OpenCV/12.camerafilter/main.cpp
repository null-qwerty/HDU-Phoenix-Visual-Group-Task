#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc,char* argv[],char** env)
{
    VideoCapture capture(0);
    Mat frame;

    while(1)
    {
        capture >> frame;
        Mat medianblur, aveblur, gaussianblur;
        int size = 7;

        medianBlur(frame, medianblur, size);
        blur(frame, aveblur, Size(size, size));
        GaussianBlur(frame, gaussianblur, Size(size, size), 0);

        namedWindow("中值滤波", WINDOW_NORMAL);
        namedWindow("均值滤波", WINDOW_NORMAL);
        namedWindow("高斯滤波", WINDOW_NORMAL);
        imshow("中值滤波", medianblur);
        imshow("均值滤波", aveblur);
        imshow("高斯滤波", gaussianblur);
        
        waitKey(30);
        if (getWindowProperty("中值滤波", WND_PROP_AUTOSIZE) != 0 ||
            getWindowProperty("均值滤波", WND_PROP_AUTOSIZE) != 0 ||
            getWindowProperty("高斯滤波", WND_PROP_AUTOSIZE) != 0)
            break;
    }

    capture.release();
    destroyAllWindows();
    return 0;
}