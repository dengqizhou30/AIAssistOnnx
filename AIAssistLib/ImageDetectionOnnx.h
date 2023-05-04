#pragma once
#include<string>
#include<iostream>
#include<fstream>
#include<cmath>
#include <windows.h>
#include <Tlhelp32.h>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>

#include "AssistConfig.h"
#include "YOLOv6.h"

using namespace cv;
using namespace std;


//定义检测结果结构体
/*
typedef struct DetectResults
{
    std::vector< int > classIds;
    std::vector< float > confidences;
    std::vector< Rect > boxes;
    int maxPersonConfidencePos; //人类 最大执行度
} DETECTRESULTS; */

// 基于opencv dnn的图像检测类
// 图像对象检查封装类，使用AI模型截取屏幕对象进行对象检测
class ImageDetectionOnnx
{
public:
    ImageDetectionOnnx();
    ~ImageDetectionOnnx();
    void ReInit();

    void getScreenshot();
    DETECTRESULTS detectImg();
    cv::Mat getImg();

private:
    void initImg(); //初始化图像检测资源
    void releaseImg(); //释放图像检测资源

private:
    static AssistConfig* m_AssistConfig;

    //截屏图像数据相关属性
    HDC m_screenDC;
    HDC m_memDC;
    HBITMAP m_hBitmap;
    BITMAPINFO* m_Bitmapinfo = NULL;

    //存放图像数据的mat对象
    //注意opencv的函数使用这个对象时会自动调用create函数创建数据存放数组
    //在其他地方使用时，需要手动调用create申请内存创建数组；重复执行create函数不会导致重复创建数据存放数组；
    cv::Mat m_mat_src; //存放bitmap中获取的源图
    cv::Mat m_mat_3; //存放转换为3通道的图像

    //AI网络相关属性
    //const string ModelFile = "./model/yolov6n-320x320.onnx";
    //const string ModelFile = "../../Data/model/onnx/yolov6n-320x320.onnx";
    const string ModelFile = "./model/onnx/yolov6n-320x320.onnx";
    //const string ModelFile = "../../Data/model/onnx/yolov6n.onnx";
    //const string ModelFile = "../../Data/model/onnx/yolov6n-0.2.onnx";
    const float MinConfidence = 0.50; //最小置信度
    const int PersonClassId = 1; //分类标签列表中 人类 的classid
    
    //使用专门的对象检测模型类
    onnxlib::YOLOv6* yolov6 = NULL ;

};






