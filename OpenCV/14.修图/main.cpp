/* 直接使用修图软件（大雾） */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

#define Filename1 "../src/14-1.jpg"
#define Filename2 "../src/14-2.jpg"
#define Result1 "../result/14-1.jpg"
#define Result2 "../result/14-2.jpg"

void Duang(Mat& src,Mat& output,double rate)//duang~的一声就完成了(不是)
{
    Mat bulr;
    bilateralFilter(src, bulr, 30, 75, 75); // 使用双边滤波实现磨皮效果
    addWeighted(src, 1 - rate, bulr, rate, 0, output); // 处理后的图与原图加权求和,减少处理后的失真
}

int main(int argc,char* argv[],char** env)
{
    Mat srcImg1 = imread(Filename1);
    Mat srcImg2 = imread(Filename2);
    Mat Output1, Output2;

    imshow("14-1 before", srcImg1);
    imshow("14-2 before", srcImg2);

    Duang(srcImg1, Output1, 0.85);
    Duang(srcImg2, Output2, 0.7);

    imshow("14-1 after", Output1);
    imshow("14-2 after", Output2);
    imwrite(Result1, Output1);
    imwrite(Result2, Output2);

    waitKey(0);
    return 0;
}