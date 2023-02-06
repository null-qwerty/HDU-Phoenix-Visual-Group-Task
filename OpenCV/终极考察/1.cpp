#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char *argv[], char **env)
{
    Mat srcImg = imread("../src/终极考察1.jpg");
    Mat gray, binary;

    Mat imageROI(srcImg, Rect(srcImg.cols / 2, 20, srcImg.cols / 2, 3 * srcImg.rows / 4));
    imshow("ROI1", imageROI);

    cvtColor(srcImg, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 0, 255, THRESH_OTSU); // 自动设置阈值
    //imshow("binary", binary);

    vector<vector<Point>> contours;                               // 连通区域轮廓点集
    findContours(binary, contours, RETR_LIST, CHAIN_APPROX_NONE); // 找连通区域,获取全部轮廓的全部像素

    for (int i = 0; i < contours.size(); i++)
    {
        Rect boundingbox = boundingRect(contours[i]);
        if (boundingbox.width * 1.0 / boundingbox.height < 1.1 &&
            boundingbox.width * 1.0 / boundingbox.height > 0.9 && boundingbox.area() > 50) // 找需要的连通区域,圆外接正方形
            drawContours(srcImg, contours, i, Scalar(0, 255, 255), -1);                 // 填充颜色
    }

    imshow("output1", srcImg);
    imwrite("../result/1.jpg",srcImg);

    waitKey(0);
    return 0;
}