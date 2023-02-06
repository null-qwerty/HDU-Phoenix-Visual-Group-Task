#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc,char* argv[],char** env)
{
    Mat srcImg = imread("../src/终极考察2.jpg");
    Mat gray, binary;

    Mat imageROI(srcImg, Rect(srcImg.cols / 2 - 100, srcImg.rows / 2 - 100, 200, 200));
    imshow("ROI2", imageROI);

    cvtColor(srcImg, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 174, 255, 0);
    //imshow("binary", binary);

    vector<vector<Point>> contours; // 连通区域轮廓点集
    findContours(binary, contours, RETR_LIST, CHAIN_APPROX_NONE); // 找连通区域,获取全部轮廓的全部像素

    for (int i = 0; i < contours.size(); i++)
    {
        RotatedRect rbox = minAreaRect(contours[i]); //minAreaRect获取的外接四边形可旋转
        if (rbox.size.width * 1.0 / rbox.size.height < 1.1 &&
                rbox.size.width * 1.0 / rbox.size.height > 0.9 && rbox.size.area() > 20) // 中间正方形
            {
                drawContours(srcImg, contours, i, Scalar(0, 255, 255)); //边界线
                Point2f corner[4];
                rbox.points(corner);
                for (int j = 0; j < 4; j++)
                    line(srcImg, corner[j], corner[j + 1 == 4 ? 0 : j + 1], Scalar(0, 0, 255), 2);  //外接矩形
            }
    }
    
    imshow("output2", srcImg);
    imwrite("../result/2.jpg",srcImg);

    waitKey(0);
    return 0;
}