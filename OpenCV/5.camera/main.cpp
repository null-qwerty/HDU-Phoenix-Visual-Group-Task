#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc,char* argv[],char** env)
{
    VideoCapture cap(0);    //读取摄像头
    Mat frame;
    if(!cap.isOpened())
    {
        cout << "视频读取失败" << endl;
        return -1;
    }
    cout << "视频读取成功" << endl;

    char str[10];   //存储包含fps的字符串
    string fpsstring;   //显示的字符串
    double t;   //计算fps的变量
    while (1)
    {
        t = (double)getTickCount();
        cap >> frame;

        t = ((double)getTickCount() - t) / getTickFrequency();  //两次读取图片的tick数 / 每秒tick数再取倒数即为fps
        double fps = 1.0 / t;
        sprintf(str, "%.3lf", fps);
        fpsstring = "FPS:";
        fpsstring += str;
        //fps输出到窗口
        putText(frame,  //目标图片 
                fpsstring, //显示的字符串
                Point(5, 20), //显示区域位置左下角的点
                FONT_HERSHEY_SIMPLEX, //字体类型
                0.5, //字体大小
                Scalar(0, 0, 0) //字体颜色
                );
        
        namedWindow("camera", 0);
        imshow("camera", frame);
        waitKey(30);
        if (getWindowProperty("camera", WND_PROP_AUTOSIZE) != 0)
            break;
    }
    
    cap.release();
    destroyAllWindows();
    return 0;
}