#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

#define BackgroundNum 1

void BoundingBox(Mat &srcImg, Mat &trans, int &number);

int main(int argc, char *argv[], char **env)
{
    Mat srcImg = imread("../11.png");
    Mat trans;
    imshow("原图", srcImg);

    threshold(srcImg, trans, 100, 255, 1); // 类型为1,二值化后取反
    cvtColor(trans, trans, COLOR_BGR2GRAY);
    imshow("二值化", trans);

    int number;
    BoundingBox(srcImg, trans, number);

    //cout << "num:" << number - BackgroundNum << endl;
    putText(srcImg, format("Total:%d", number - BackgroundNum), Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2, 8);
    imshow("output", srcImg);
    imwrite("../result.png", srcImg);

    waitKey(0);
    return 0;
}
void BoundingBox(Mat &srcImg, Mat &trans, int &number)
{
    Mat stats, centroids;
    vector<Vec3b> colors;
    RNG rng(time(NULL));
    number = connectedComponentsWithStats(trans, trans, stats, centroids, 8, CV_16U);
    
    for (int i = 0; i < number; i++) // 随机颜色
    {
        Vec3b color = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        colors.push_back(color);
    }

    for (int i = 1; i < number; i++)
    {
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);
        Rect rect(x, y, w, h); // 获取连通区域外接矩形

        rectangle(srcImg, rect, colors[i], 2, 8); // 画矩形
        putText(srcImg, format("%d", i), Point(x + 5, y + 20), FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2); // 打印文字到图片
    }
}