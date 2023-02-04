#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

void AverageRGB(Mat &InputImage, Mat &OutputImage)
{
    OutputImage = InputImage.clone();   //拷贝副本
    int row = OutputImage.rows;         //处理行数
    int col = OutputImage.cols * OutputImage.channels();    //处理列数,等于图的列数*通道数

    for (int i = 0; i < row; i++)
    {
        uchar *data = OutputImage.ptr<uchar>(i);
        for (int j = 0; j < col; j += 3)
        {
            uchar ave = (data[j] + data[j + 1] + data[j + 2]) / 3;  //RGB平均值
            data[j] = data[j + 1] = data[j + 2] = ave;      //将RGB替换为平均值
        }
    }
}
int main(int argc, char *argv[], char **env)
{
    Mat InputImage = imread("../image.jpg");
    Mat OutputImage;

    AverageRGB(InputImage, OutputImage); //遍历input,处理像素存储到output

    imshow("Output", OutputImage);  //结果为转成灰度图

    waitKey(0);
}