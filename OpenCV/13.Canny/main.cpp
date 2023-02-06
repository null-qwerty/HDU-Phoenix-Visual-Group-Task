#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc,char* argv[],char** env)
{
    Mat srcImg = imread("../image.jpg");
    Mat gray, Output;
    imshow("src", srcImg);

    cvtColor(srcImg, gray, COLOR_BGR2GRAY);  // 第一步:转成灰度图
    GaussianBlur(gray, gray, Size(3, 3), 0); // 第二步:使用滤波降噪
                                             // 滤波会抹除部分细节,导致最终边缘线条减少,但可以减少Canny算子的计算量

    Canny(gray, Output, 200, 100, 3); // Canny(输入图片,输出图片,阈值1,阈值2,Sobel算子大小);
                                      // 两个阈值自动判断大小,Sobel算子大小为>=3&&<=7的奇数

    imshow("edge", Output);
    imwrite("../result.jpg", Output);

    waitKey(0);
    return 0;
}