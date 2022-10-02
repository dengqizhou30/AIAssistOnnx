// OnnxTest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <windows.h>
#include <time.h>

#include<opencv.hpp>

#include "YOLOv6.h"

using namespace cv;
using namespace std;

int main()
{
    std::cout << "Hello World!\n";

    HDC m_screenDC;
    HDC m_memDC;
    HBITMAP m_hBitmap;

    int width = 320, height = 320;

    // 获取屏幕 DC
    m_screenDC = GetDC(HWND_DESKTOP);
    m_memDC = CreateCompatibleDC(m_screenDC);
    // 创建位图
    m_hBitmap = CreateCompatibleBitmap(m_screenDC, width, height);
    SelectObject(m_memDC, m_hBitmap);


    cv::Mat mat1, mat2;
    mat1.create(height, width, CV_8UC4);
    //mat.create(height, width, CV_32FC4);


    double duration1;
    clock_t start1, finish1;

    start1 = clock();


    // 得到位图的数据
    // 使用BitBlt截图，性能较低，后续修改为DXGI
    //Windows8以后微软引入了一套新的接口，叫“Desktop Duplication API”，应用程序，可以通过这套API访问桌面数据。
    //由于Desktop Duplication API是通过Microsoft DirectX Graphics Infrastructure (DXGI)来提供桌面图像的，速度非常快。
    bool opt = BitBlt(m_memDC, 0, 0, width, height, m_screenDC, 300, 250, SRCCOPY);

    int iBits = GetDeviceCaps(m_memDC, BITSPIXEL) * GetDeviceCaps(m_memDC, PLANES);
    WORD wBitCount;
    if (iBits <= 1)
        wBitCount = 1;
    else if (iBits <= 4)
        wBitCount = 4;
    else if (iBits <= 8)
        wBitCount = 8;
    else if (iBits <= 24)
        wBitCount = 24;
    else
        wBitCount = 32;
    BITMAPINFOHEADER bi = { sizeof(bi), width, -height, 1, wBitCount, BI_RGB };


    //int rows = GetDIBits(m_screenDC, m_hBitmap, 0, detectRect.height, m_mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    int rows = GetDIBits(m_memDC, m_hBitmap, 0, height, mat1.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    //cv::namedWindow("opencvwindows", WINDOW_AUTOSIZE);
    //cv::imshow("opencvwindows", mat1);
    //cout << "show GetDIBits mat\n";
    //waitKey(10000);


    //屏幕截图和视频截图格式不一样，需要进行图像格式转换
    //去掉alpha 值(透明度)通道，转换为CV_8UC3格式
    cv::cvtColor(mat1, mat2, COLOR_RGBA2RGB);

    finish1 = clock();
    duration1 = (double)(finish1 - start1) * 1000 / CLOCKS_PER_SEC;

    //std::cout << "mat.type() = " << mat.type() << std::endl;
    //std::cout << "mat2.type() = " << mat2.type() << std::endl;

    //检测图像格式
    //std::cout << "mat格式：" << mat.depth() << "," << mat.type() << "," << mat.channels() << "," << std::endl;
    //std::cout << "mat2格式：" << mat2.depth() << "," << mat2.type() << "," << mat2.channels() << "," << std::endl;

    cv::imshow("opencvwindows", mat2);
    cout << "show cvtColor mat\n";
    waitKey(2000);



    //onnx推理
    std::string onnx_path = "./model/yolov6n-320x320.onnx";

    // 1. Test Default Engine ONNXRuntime
    onnxlib::YOLOv6* yolov6 = new onnxlib::YOLOv6(onnx_path,4); // default

    std::vector<lite::types::Boxf> detected_boxes;
    //cv::Mat img_bgr = cv::imread(test_img_path);

    for (int i = 0; i < 100; i++) {
        detected_boxes.clear();
        start1 = clock();
        yolov6->detect(mat2, detected_boxes, 0.46);
        finish1 = clock();
        duration1 = (double)(finish1 - start1) * 1000 / CLOCKS_PER_SEC;
        std::cout << "检测时间: " << duration1 << ", 检查到对象个数: " << detected_boxes.size() << std::endl;
    }

    lite::utils::draw_boxes_inplace(mat2, detected_boxes);

    std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

    delete yolov6;

    cv::imshow("opencvwindows", mat2);
    cout << "show detected mat\n";
    waitKey(0);
}
