#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char *argv[], char **env)
{
    Mat srcMat=imread("../8.png");
    Mat hsvMat, Output;

    cvtColor(srcMat, hsvMat, COLOR_BGR2HSV);    //cvtColor转换色彩空间
    inRange(hsvMat, Scalar(0, 43, 46), Scalar(10, 255, 255), Output);

    namedWindow("src", 0);
    imshow("src", srcMat);
    namedWindow("Output", 0);
    imshow("Output", Output);

    waitKey(0);
    return 0;
}
/*
        黑      灰      白      红      橙      黄      绿      青      蓝      紫
hmin     0       0       0      0     11      26      35      78     100     125
hmax   180     180     180     10     25      34      77      99     124     155
smin     0       0       0     43     43      43      43      43      43      43
smax   255      43      30    255    255     255     255     255     255     255
vmin     0      46     221     46     46      46      46      46      46      46
vmax    46     220     255    255    255     255     255     255     255     255
*/