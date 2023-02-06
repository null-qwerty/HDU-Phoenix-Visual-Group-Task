#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc,char* argv[],char** env)
{
    Mat srcImg = imread("../src/终极考察3.jpg");
    vector<Mat> channel;
    Mat srcCopy=srcImg.clone();
    Mat imageROI(srcImg, Rect(srcImg.cols / 2 - 100, srcImg.rows / 2 - 100, 200, 200));
    imshow("ROI3", imageROI);

    cvtColor(srcImg, srcImg, COLOR_BGR2HSV); // BGR无法区分,转成HSV
    split(srcImg, channel);

    // for (int i = 0; i < 3; i++) // 0,1,2对应HSV,S区分明显
    //     imshow(format("%d", i), channel[i]);
    Mat s = channel[1];
    Mat binary;
    threshold(s, binary, 0, 255, THRESH_OTSU);
//    imshow("s_binary", binary);

    RNG rng(time(NULL));
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_LIST, CHAIN_APPROX_NONE);

    for (int i = 0; i < contours.size(); i++)
    {
        RotatedRect rbox = minAreaRect(contours[i]);
        if (rbox.size.area() > 500)
        {
            Point2f corner[4];
            Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            rbox.points(corner);
            for (int j = 0; j < 4; j++)
                line(srcCopy, corner[j], corner[j + 1 == 4 ? 0 : j + 1], color, 2);
        }
    }

    imshow("output3", srcCopy);
    imwrite("../result/3.jpg",srcCopy);

    waitKey(0);
    return 0;
}