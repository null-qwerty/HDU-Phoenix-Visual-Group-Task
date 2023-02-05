/*
 * gamma校正起源于人的感知亮度与实际物理亮度的差异,其转换公式为
 *                   Vout=Vin^gamma
 * 其中Vout指感知亮度(输出),Vin指物理亮度(输入),gamma一般取值2.2
 * 通过gamma校正能使原图阴影部分的细节明显
 */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>
#include <iostream>

using namespace cv;
using namespace std;

#define OPENFILENAME1 "../src/7-1.png"
#define OPENFILENAME2 "../src/7-2.jpg"

void GammaCorrection(Mat &src, Mat &OutputImg, double gamma = 2.2)
{
    OutputImg = src.clone();
    int row = OutputImg.rows;
    int col = OutputImg.cols;

    for (int i = 0; i < row; i++)
    {
        uchar *data = OutputImg.ptr<uchar>(i);
        for (int j = 0; j < col; j++)
            data[j] = pow(data[j] / 255.0, 1.0 / gamma) * 255; //先将颜色数值从0~255映射到0~1,公式转换完成后转回0~255
    }
}
int main()
{
    Mat img1 = imread(OPENFILENAME1,0); //以灰度图读入
    Mat img2 = imread(OPENFILENAME2, 0);
    Mat Output1, Output2;

    namedWindow("7-1", 0);
    imshow("7-1", img1);
    namedWindow("7-2", 0);
    imshow("7-2", img2);

    GammaCorrection(img1, Output1);
    GammaCorrection(img2, Output2);

    namedWindow("7-1result", 0);
    imshow("7-1result", Output1);
    namedWindow("7-2result", 0);
    imshow("7-2result", Output2);
    
    waitKey(0);
    return 0;
}