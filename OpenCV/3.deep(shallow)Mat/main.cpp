#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

void AverageRGB(Mat& inputImage)
{
    int row = inputImage.rows;
    int col = inputImage.cols * inputImage.channels();

    for (int i = 0; i < row; i++)
    {
        uchar *data = inputImage.ptr<uchar>(i);
        uchar threshold = 100;
        for (int j = 0; j < col; j += 3)
        {
            uchar ave = (data[j] + data[j + 1] + data[j + 2]) / 3;
            ave = ave > threshold ? 255 : 0;
            data[j] = data[j + 1] = data[j + 2] = ave;
        }
    }
}
int main(int argc, char *argv[], char **env)
{
    Mat srcMat = imread("../image.jpg");
    Mat deepMat, shallowMat;

    srcMat.copyTo(deepMat); //深复制,近似于deepMat = srcMat.clone();
    shallowMat = srcMat;    //浅复制

    imshow("deepMat before change", deepMat);
    imshow("shallowMat before change", shallowMat);

    AverageRGB(srcMat);

    imshow("deepMat after change", deepMat);
    imshow("shallowMat after change", shallowMat);
//深复制在内存中分配新的空间保存复制的图片，对原图片的修改不影响deepMat
//浅复制不分配新的空间保存，对原图的修改就是对shallowMat的修改
    waitKey(0);
}