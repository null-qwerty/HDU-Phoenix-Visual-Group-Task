#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

#define BackgroundNum 1

int main(int argc,char* argv[],char** env)
{
    Mat srcImg = imread("../10.png");
    Mat trans;
    imshow("src", srcImg);

    threshold(srcImg, trans, 85, 255, 0); // 二值化,阈值85,类型一般取0
    // threshold(input,output,阈值,大于阈值的取值,类型);
    cvtColor(trans, trans, COLOR_BGR2GRAY); // 结果转灰度

    //int number = connectedComponents(Output, result,8,CV_16U);
    Mat stats, centroids;
    vector<Vec3b> colors;
    RNG rng(time(NULL)); // RNG是OpenCV自带的随机数类
    // RNG rng(随机数种);
    int number = connectedComponentsWithStats(trans, trans, stats, centroids, 8, CV_16U); // 统计连通区域数量，withstats可保存连通区域的位置性质
                                                                                          //  stats保存左上角坐标和长宽,centroids保存重心坐标
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
        // rectangle(图片, 矩形, 颜色, 线宽, 线型);
        putText(srcImg, format("%d", i), Point(x + 5, y + 20), FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2); // 打印文字到图片
        // putText(图片, 输出字符串, 左上角点, 类型, 字体大小, 颜色, 线宽);
    }

    imshow("output", srcImg);
    imwrite("../result.png", srcImg);
    cout << "Coin num:" << number - BackgroundNum << endl; // 硬币数 = 总连通区域 - 背景

    waitKey(0);
    return 0;
}