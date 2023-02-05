#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

#define WINDOW_SIZE 800

void DrawCircle(Mat background,Point Center,int Radius)
{
    int thickness = 4;
    int linetype = 8;

    circle(background,  //背景图
           Center,      //圆心
           Radius,      //半径
           Scalar(0, 0, 255),   //颜色
           thickness,   //线段粗细,-1表示填充
           linetype);   //线型
}
void DrawLine(Mat background,Point StartPoint,Point EndPoint)
{
    int thickness = 4;
    int linetype = 8;

    line(background,
         StartPoint,    //起始点
         EndPoint,      //终点
         Scalar(0, 0, 255),
         thickness,
         linetype);
}
void DrawRect(Mat background,Point TopLeftPoint,Point BottomRightPoint)
{
    int thickness = 4;
    int linetype = 8;

    rectangle(background,
              TopLeftPoint,     //左上角
              BottomRightPoint, //右下角
              Scalar(0, 0, 255),
              thickness,
              linetype);
}
int main(int argc,char* argv[],char** env)
{
    Mat img = Mat::zeros(WINDOW_SIZE, WINDOW_SIZE, CV_8UC3);

    DrawCircle(img, Point(WINDOW_SIZE / 2, WINDOW_SIZE / 2), WINDOW_SIZE / 10);
    DrawLine(img, Point(WINDOW_SIZE / 2, WINDOW_SIZE / 8), Point(WINDOW_SIZE / 2, WINDOW_SIZE - WINDOW_SIZE / 8));
    DrawLine(img, Point(WINDOW_SIZE / 8, WINDOW_SIZE / 2), Point(WINDOW_SIZE - WINDOW_SIZE / 8, WINDOW_SIZE / 2));
    DrawRect(img, Point(WINDOW_SIZE / 4, WINDOW_SIZE / 4), Point(WINDOW_SIZE - WINDOW_SIZE / 4, WINDOW_SIZE - WINDOW_SIZE / 4));

    namedWindow("Show", 0);
    imshow("Show", img);
    imwrite("../result.jpg", img);

    waitKey(0);
    return 0;
}