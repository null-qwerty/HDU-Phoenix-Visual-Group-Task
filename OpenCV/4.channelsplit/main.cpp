#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    Mat srcMat = imread("../image.jpg");
    vector<Mat> channel;

    split(srcMat, channel);
    //split分离通道,opencv以BGR存储像素
    namedWindow("src", 0);
    namedWindow("B", 0);
    namedWindow("G", 0);
    namedWindow("R", 0);
    imshow("src", srcMat);
    imshow("B", channel[0]);
    imshow("G", channel[1]);
    imshow("R", channel[2]);

    waitKey(0);
}