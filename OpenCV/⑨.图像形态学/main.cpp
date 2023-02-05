#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

string ElementShape[3] = {"矩形", "交叉形", "椭圆形"};

string filename(string process, int shape, int ksize)
{
    string path = "../result/" + process+"/";
    char sizestr[10];

    sprintf(sizestr, "%dx%d", ksize, ksize);
    path += ElementShape[shape] + sizestr + ".jpg";

    return path;
}
void OpenOperation(Mat &Input, Mat &Output, Mat Element)    //开运算,先腐蚀后膨胀
{
    Mat temp;
    erode(Input, temp, Element);
    dilate(temp, Output, Element);
}
void ClosingOperation(Mat &Input,Mat& Output,Mat Element)   //闭运算,先膨胀后腐蚀
{
    Mat temp;
    dilate(Input, temp, Element);
    erode(temp, Output, Element);
}
int main(int argc,char* argv[],char** env)
{
    Mat srcImg = imread("../9.png");
    Mat element;
    Mat Output;

    for (int shape = 0; shape < 3; shape++) //形状三种,0(MORPH_RECT),1(MORPH_CROSS),2(MORPH_ELLIPSE)
    {
        for (int ksize = 3; ksize <= 11; ksize += 2)
        {
            element = getStructuringElement(shape, Size(ksize, ksize)); //创建算子(核)
            
            erode(srcImg, Output, element);
            imwrite(filename("腐蚀", shape, ksize), Output);
            dilate(srcImg, Output, element);
            imwrite(filename("膨胀", shape, ksize), Output);
            OpenOperation(srcImg, Output, element);
            imwrite(filename("开运算", shape, ksize), Output);
            ClosingOperation(srcImg, Output, element);
            imwrite(filename("闭运算", shape, ksize), Output);
        }
    }
    waitKey(0);
    return 0;
}