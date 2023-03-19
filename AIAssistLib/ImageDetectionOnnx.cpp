#include "pch.h"
#include "ImageDetectionOnnx.h"
using cv::Mat;

//静态成员初始化
AssistConfig* ImageDetectionOnnx::m_AssistConfig = AssistConfig::GetInstance();


//类方法实现
ImageDetectionOnnx::ImageDetectionOnnx()
{
    initImg();

    //加载AI模型
    yolov6 = new onnxlib::YOLOv6(ModelFile, 4);
}

ImageDetectionOnnx::~ImageDetectionOnnx()
{
    //图像资源释放
    releaseImg();

    //释放资源网络
    try {
        if (yolov6 != NULL) {
            delete yolov6;
            yolov6 = NULL;
        }
    }
    catch (Exception ex) {
        string msg = "";
    }
}

//修改配置后，需要重新初始化一些对象
void ImageDetectionOnnx::ReInit() {
    releaseImg();
    initImg();
}

//初始化图像资源
void ImageDetectionOnnx::initImg(){
    //注意屏幕缩放后的情况
    //注意抓取屏幕的时候使用缩放后的物理区域坐标，抓取到的数据实际是逻辑分辨率坐标
    cv::Rect detectRect = m_AssistConfig->detectRect;
    cv::Rect detectZoomRect = m_AssistConfig->detectZoomRect;


    // 获取屏幕 DC
    m_screenDC = GetDC(HWND_DESKTOP);
    m_memDC = CreateCompatibleDC(m_screenDC);
    // 创建位图
    m_hBitmap = CreateCompatibleBitmap(m_screenDC, detectRect.width, detectRect.height);
    SelectObject(m_memDC, m_hBitmap);

    //分析位图信息头
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
    m_Bitmapinfo = new BITMAPINFO{ {sizeof(BITMAPINFOHEADER), detectRect.width, -detectRect.height, 1, wBitCount, BI_RGB },{0,0,0,0} };

    //创建存放图像数据的mat
    //m_mat.create(detectRect.height, detectRect.width, CV_8UC4);
    //m_mat3.create(detectRect.height, detectRect.width, CV_8UC3);
}

//释放图像资源
void ImageDetectionOnnx::releaseImg() {

    //资源释放
    try {
        m_mat_src.release();
        m_mat_3.release();

        if (m_Bitmapinfo != NULL)
            delete m_Bitmapinfo;
        DeleteObject(m_hBitmap);
        DeleteDC(m_memDC);
        ReleaseDC(HWND_DESKTOP, m_screenDC);
    }
    catch (Exception ex) {
        string msg = "";
    }

    m_Bitmapinfo = NULL;
    m_hBitmap = NULL;
    m_memDC = NULL;
    m_screenDC = NULL;
}


/* 获取检测区的屏幕截图 */
void ImageDetectionOnnx::getScreenshot()
{
    //计算屏幕缩放后的，裁剪后的实际图像检查区域
    //注意抓取屏幕的时候使用缩放后的物理区域坐标，抓取到的数据实际是逻辑分辨率坐标
    cv::Rect detectRect = m_AssistConfig->detectRect;
    cv::Rect detectZoomRect = m_AssistConfig->detectZoomRect;
   

    // 得到位图的数据
    // 使用BitBlt截图，性能较低，后续修改为DXGI
    //Windows8以后微软引入了一套新的接口，叫“Desktop Duplication API”，应用程序，可以通过这套API访问桌面数据。
    //由于Desktop Duplication API是通过Microsoft DirectX Graphics Infrastructure (DXGI)来提供桌面图像的，速度非常快。
    bool opt = BitBlt(m_memDC, 0, 0, detectRect.width, detectRect.height, m_screenDC, detectZoomRect.x, detectZoomRect.y, SRCCOPY);

    //注意在非opencv函数中使用mat时，需要手动调用create申请内存创建数组；重复执行create函数不会导致重复创建数据存放数组；
    m_mat_src.create(detectRect.height, detectRect.width, CV_8UC4);
    //int rows = GetDIBits(m_screenDC, m_hBitmap, 0, detectRect.height, m_mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    int rows = GetDIBits(m_memDC, m_hBitmap, 0, detectRect.height, m_mat_src.data, (BITMAPINFO*)m_Bitmapinfo, DIB_RGB_COLORS);

    //cv::namedWindow("opencvwindows", WINDOW_AUTOSIZE);
    //cv::imshow("opencvwindows", m_mat_src);
    //waitKey(2000);

    //屏幕截图和视频截图格式不一样，需要进行图像格式转换
    //去掉alpha 值(透明度)通道，转换为CV_8UC3格式
    cv::cvtColor(m_mat_src, m_mat_3, COLOR_RGBA2RGB);

    //数据格式转换
    //m_mat_3.convertTo(img_mat, CV_8UC3, 1.0);

    //根据不同的游戏做一定的特殊处理，清理掉游戏者个人图像
    //super people 和 pubg两个游戏做特殊处理
    //影响识别效果，先关闭这块逻辑
    /*
    if (m_AssistConfig->gameIndex == 0 || m_AssistConfig->gameIndex == 1) {
        int y = m_mat_3.rows * 3 / 4;
        Mat mask(m_mat_3, Rect(0, y, m_mat_3.cols/2, m_mat_3.rows- y));
        mask = Scalar(0, 0, 0);
    }*/

    return;
}


/* AI推理 */
DETECTRESULTS ImageDetectionOnnx::detectImg()
{
    //计算屏幕缩放后的，裁剪后的实际图像检查区域
    //注意抓取屏幕的时候使用缩放后的物理区域坐标，抓取到的数据实际是逻辑分辨率坐标
    cv::Rect detectRect = m_AssistConfig->detectRect;
    cv::Rect detectZoomRect = m_AssistConfig->detectZoomRect;
    int gameIndex = m_AssistConfig->gameIndex;
    int playerCentX = m_AssistConfig->playerCentX;

    DETECTRESULTS out;

    std::vector<lite::types::Boxf> detected_boxes;

    try
    {
        //执行模型推理
        yolov6->detect(m_mat_3, detected_boxes, MinConfidence);

        //处理推理结果
        for (int i = 0; i < detected_boxes.size(); i++) {
        
            //保存这个检测到的对象
            out.classIds.push_back(detected_boxes[i].label);
            out.confidences.push_back(detected_boxes[i].score);
            out.boxes.push_back(detected_boxes[i].rect());

            //保存置信度最大的人员的位置
            if (i == 0) {
                out.maxPersonConfidencePos = 0;
            }
              
        }

        //使用计算好的游戏屏幕中心坐标
        LONG x1 = m_AssistConfig->detectCentX;
        LONG y1 = m_AssistConfig->detectCentY;

        //调整选择检测到的目标的逻辑，除了考虑啊最大置信度外，还要考虑距离的远近
        float maxvalue = 0;
        for (int i = 0; i < out.boxes.size(); i++) {
            float confidence = out.confidences.at(i);
            Rect box = out.boxes.at(i);

            //计算检测到对象的中心坐标，计算为靠上的位置，尽量打头
            LONG x2 = m_AssistConfig->detectRect.x + box.x + box.width / 2;
            LONG y2 = m_AssistConfig->detectRect.y + box.y + box.height / 3;

            //计算距离因素
            float xval = 0.0;
            if (abs(x2 - x1) < m_AssistConfig->detectRect.width / 2)
                xval = (float)(m_AssistConfig->detectRect.width / 2 - abs(x2 - x1)) / m_AssistConfig->detectRect.width;
            float yval = 0.0;
            if(abs(y2 - y1) < m_AssistConfig->detectRect.height / 2)
                yval = (float)(m_AssistConfig->detectRect.height / 2 - abs(y2 - y1)) / m_AssistConfig->detectRect.height;
            out.xvals.push_back(xval);
            out.yvals.push_back(yval);

            //置信度和距离因素合并，距离因素占比小于0.5
            float value = confidence + 0.5 * xval + 0.5 * yval;
            if (value > maxvalue) {
                maxvalue = value;
                out.maxPersonConfidencePos = i;
            }
        }
    }
    catch (Exception ex) {
        string msg = "";
    }

    return out;
}


cv::Mat ImageDetectionOnnx::getImg() {
    //克隆mat数据结外部程序使用，这个类自身只重用两个mat对象及他们的数据内存区
    Mat mat = m_mat_3.clone();
    return mat;
}