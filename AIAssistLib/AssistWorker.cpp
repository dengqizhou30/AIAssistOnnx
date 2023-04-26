#include "pch.h"
#include "AssistWorker.h"


//初始化静态成员变量
AssistConfig* AssistWorker::m_AssistConfig = AssistConfig::GetInstance();
std::condition_variable AssistWorker::m_pushCondition = std::condition_variable();
std::atomic_bool AssistWorker::m_startPush = false;   ///是否满足压枪条件标志
std::atomic_int AssistWorker::m_pushCount = 0;
WEAPONINFO AssistWorker::m_weaponInfo = { 1,1,1 };

std::atomic_bool AssistWorker::m_startFire = false;   ///是否正在开枪，避免正在人工开枪时再执行自动开枪操作



//定义鼠标钩子函数
LRESULT CALLBACK MouseHookProcedure(int nCode, WPARAM wParam, LPARAM lParam)
{
    MSLLHOOKSTRUCT* p = (MSLLHOOKSTRUCT*)lParam;
    BOOLEAN injected = p->flags & LLMHF_INJECTED || p->flags & LLMHF_LOWER_IL_INJECTED; // Checks if click was injected and not from a mouse
    if (nCode == HC_ACTION && !injected)
    {
        if (wParam == WM_LBUTTONDOWN) {
            //设置正在开枪标志
            AssistWorker::m_startFire = true;

            //判断用户是否设置了自动压枪
            if (AssistWorker::m_AssistConfig->autoPush) {
                //开始压枪
                AssistWorker::m_startPush = true;
                AssistWorker::m_pushCount = 0;
                AssistWorker::m_pushCondition.notify_all();       
            }
        }
        else if (wParam == WM_LBUTTONUP) {
            //设置正在开枪标志
            AssistWorker::m_startFire = false;

            //判断用户是否设置了自动压枪
            if (AssistWorker::m_AssistConfig->autoPush) {
                //鼠标左键抬起后结束压枪
                AssistWorker::m_startPush = false;
                AssistWorker::m_pushCount = 0;
                AssistWorker::m_pushCondition.notify_all();
            }
        }
        else if (wParam == WM_RBUTTONDOWN) {
        }
        else if (wParam == WM_RBUTTONUP) {
        }
        else if (wParam == WM_MBUTTONDOWN) {
        }
        else if (wParam == WM_MBUTTONUP) {
        }
    }
    return CallNextHookEx(nullptr, nCode, wParam, lParam);
}

//定义键盘钩子函数
LRESULT CALLBACK KeyboardHookProcedure(int nCode, WPARAM wParam, LPARAM lParam)
{
    // WH_KEYBOARD_LL uses the LowLevelKeyboardProc Call Back
    // wParam and lParam parameters contain information about the message.
    auto* p = (KBDLLHOOKSTRUCT*)lParam;
    if (nCode == HC_ACTION)
    {
        if (wParam == WM_SYSKEYDOWN || wParam == WM_KEYDOWN)
        {
            //检查是否切换背包
            switch (p->vkCode) {
                case 49: 
                    AssistWorker::m_weaponInfo.bag = 1;
                    break;
                case 50:
                    AssistWorker::m_weaponInfo.bag = 2;
                    break;
                case 51:
                    AssistWorker::m_weaponInfo.bag = 3;
                    break;
                case 52:
                    AssistWorker::m_weaponInfo.bag = 4;
                    break;
                case 53:
                    AssistWorker::m_weaponInfo.bag = 5;
                    break;

                case VK_NUMPAD1:
                    MouseKeyboard::m_AssistConfig->maxJuPushCount = 6;
                    break;
                case VK_NUMPAD2:
                    MouseKeyboard::m_AssistConfig->maxJuPushCount = 12;
                    break;
                case VK_NUMPAD3:
                    MouseKeyboard::m_AssistConfig->maxJuPushCount = 18;
                    break;
                case VK_NUMPAD0:
                    MouseKeyboard::m_AssistConfig->maxJuPushCount = MouseKeyboard::m_AssistConfig->maxBuPushCount;
                    break;

                case 0x56:
                    //使用v键关闭/开启自动追踪、开火、压枪
                    if (AssistWorker::m_AssistConfig->autoTrace || AssistWorker::m_AssistConfig->autoFire || AssistWorker::m_AssistConfig->autoPush) {
                        AssistWorker::m_AssistConfig->autoTrace = false;
                        AssistWorker::m_AssistConfig->autoFire = false;
                        AssistWorker::m_AssistConfig->autoPush = false;
                    }
                    else {
                        //恢复用户设置的值
                        AssistWorker::m_AssistConfig->autoTrace = AssistWorker::m_AssistConfig->autoTraceUserSet;
                        AssistWorker::m_AssistConfig->autoFire = AssistWorker::m_AssistConfig->autoFireUserSet;
                        AssistWorker::m_AssistConfig->autoPush = AssistWorker::m_AssistConfig->autoPushUserSet;

                    }
                    break;

                case VK_ESCAPE:
                    //使用esc键关闭自动追踪、开火、压枪
                    if (AssistWorker::m_AssistConfig->autoTrace || AssistWorker::m_AssistConfig->autoFire || AssistWorker::m_AssistConfig->autoPush) {

                        AssistWorker::m_AssistConfig->autoTrace = false;
                        AssistWorker::m_AssistConfig->autoFire = false;
                        AssistWorker::m_AssistConfig->autoPush = false;

                    }
                   
            }
        }
    }
    return CallNextHookEx(nullptr, nCode, wParam, lParam);
}


AssistWorker::AssistWorker()
{
    //先重新计算检测区域相关数据
    m_AssistConfig->ReCalDetectionRect();

    //创建队列
    //鼠标操作队列长度设置为1，目的是只处理最新检测结果。
    fireQueue = new BlockQueue<DETECTRESULTS>(1);
    moveQueue = new BlockQueue<DETECTRESULTS>(1);

    drawQueue = new BlockQueue<DRAWRESULTS>(10);
    outDrawQueue = new BlockQueue<Mat>(10);

    //创建工作线程
    //图片检测线程
    detectThread = new thread(std::bind(&AssistWorker::DetectWork, this));
    detectThread->detach();
    //自动开火线程
    fireThread = new thread(std::bind(&AssistWorker::FireWork, this));
    fireThread->detach();
    //自动瞄准
    moveThread = new thread(std::bind(&AssistWorker::MoveWork, this));
    moveThread->detach();
    //绘图线程
    drawThread = new thread(std::bind(&AssistWorker::DrawWork, this));
    drawThread->detach();

    //drawAimThread = new thread(std::bind(&AssistWorker::DrawAimWork, this));
    //drawAimThread->detach();
    
    //鼠标键盘hook线程,
    mouseKeyboardHookThread = new thread(std::bind(&AssistWorker::MouseKeyboardHookWork, this));
    mouseKeyboardHookThread->detach();
    
    //压枪线程
    pushThread = new thread(std::bind(&AssistWorker::PushWork, this));
    pushThread->detach();

    //创建图片检测和鼠标操作对象
    //imageDetection = new ImageDetection();
    //imageDetection = new ImageDetectionTensorflow();
    imageDetection = NULL;
    mouseKeyboard = new MouseKeyboard();
    drawImage = new DrawImage();

    srand(time(nullptr)); // 用当前时间作为种子

    return;
}

AssistWorker::~AssistWorker()
{
    m_stopFlag = true;
    
    if (imageDetection != NULL)
        delete imageDetection;
    if (mouseKeyboard != NULL)
        delete mouseKeyboard;
    if (drawImage != NULL)
        delete drawImage;

    if (drawQueue != NULL)
        delete drawQueue;
    if (fireQueue != NULL)
        delete fireQueue;
    if (moveQueue != NULL)
        delete moveQueue;

    if (detectThread != NULL)
        delete detectThread;
    if (fireThread != NULL)
        delete fireThread;
    if (moveThread != NULL)
        delete moveThread;
    if (drawThread != NULL)
        delete drawThread;
    if (drawAimThread != NULL)
        delete drawAimThread;
    if (mouseKeyboardHookThread != NULL)
        delete mouseKeyboardHookThread;
    if (pushThread != NULL)
        delete pushThread;

    if (m_mouseHook) { 
        UnhookWindowsHookEx(m_mouseHook); 
        m_mouseHook = NULL;
    }
    if (m_keyboardHook) { 
        UnhookWindowsHookEx(m_keyboardHook); 
        m_keyboardHook = NULL;
    }

    return;
}

//修改配置后，需要重新初始化一些对象
void AssistWorker::ReInit() {
    
    //先停止所有工作线程
    Pause();
    Sleep(1000);

    //先重新计算检测区域相关数据
    m_AssistConfig->ReCalDetectionRect();

    //清空所有队列
    drawQueue->Clear();
    outDrawQueue->Clear();
    fireQueue->Clear();
    moveQueue->Clear();

    //重建需要重建的对象
    if (m_AssistConfig->detectImg) {
        if (imageDetection != NULL) {
            imageDetection->ReInit();
        }
        else {
            //新建对象
            imageDetection = new ImageDetectionOnnx();
        }
    }

    //设置线程标志重启工作线程
    Start();

    return;
}

void AssistWorker::DetectWork()
{
    while (!m_stopFlag)
    {
        if (m_detectPauseFlag)
        {
            //根据线程标志进行线程阻塞
            unique_lock<mutex> locker(m_detectMutex);
            while (m_detectPauseFlag)
            {
                m_detectCondition.wait(locker); // Unlock _mutex and wait to be notified
            }
            locker.unlock();
        }
        else if(imageDetection != NULL){
            //图像检测
            double duration;
            clock_t start, finish;
            start = clock();

            //屏幕截屏图像检测
            imageDetection->getScreenshot();
            DETECTRESULTS detectResult = imageDetection->detectImg();

            finish = clock();
            duration = (double)(finish - start) * 1000 / CLOCKS_PER_SEC;

            if (detectResult.classIds.size() > 0) {
                //有检查到人类，结果放到队列中,并通知处理线程消费检测结果
                //如果把开枪和鼠标移动操作放在不同线程，会导致操作割裂，这两个操作先放回同一个线程处理
                //fireQueue->PushBackForce(detectResult);
                moveQueue->PushBackForce(detectResult);
            }

            //然后复制mat对象，用于前端显示
            Mat mat = imageDetection->getImg();

            DRAWRESULTS  drawResult{detectResult, mat, duration};
            bool push = drawQueue->PushBack(drawResult);

            //如果队列满没有推送成功，则手工释放clone的mat对象
            if (!push) {
                mat.release();
                mat = NULL;
            }
            
        }
        else {
            Sleep(500);
        }
    }

    return;
}

void AssistWorker::FireWork()
{
    while (!m_stopFlag)
    {
        if (m_firePauseFlag)
        {
            //根据线程标志进行线程阻塞
            unique_lock<mutex> locker(m_fireMutex);
            while (m_firePauseFlag)
            {
                m_fireCondition.wait(locker); // Unlock _mutex and wait to be notified
            }
            locker.unlock();
        }
        else {
            
            //获取队列中存放的检测结果
            DETECTRESULTS detectResult;
            bool ret = fireQueue->PopFront(detectResult);
            if (ret) {
                //执行自动开枪操作
                //先检查是否设置了自动开枪标志
                if (m_AssistConfig->autoFire && !AssistWorker::m_startFire) {
                    //在检查是否已经瞄准了
                    bool isInTarget = mouseKeyboard->IsInTarget(detectResult);
                    //如果已经瞄准，执行自动开枪操作
                    if (isInTarget) {
                        //开枪和鼠标移动操作放在不同线程，导致操作割裂，先放回同一个线程处理
                        //mouseKeyboard->AutoFire(detectResult);
                    }
                }
            }
            
        }
    }

    return;
}

void AssistWorker::MoveWork()
{
    while (!m_stopFlag)
    {
        if (m_movePauseFlag)
        {
            //根据线程标志进行线程阻塞
            unique_lock<mutex> locker(m_moveMutex);
            while (m_movePauseFlag)
            {
                m_moveCondition.wait(locker); // Unlock _mutex and wait to be notified
            }
            locker.unlock();
        }
        else {
            //获取队列中存放的检测结果
            DETECTRESULTS detectResult;
            bool ret = moveQueue->PopFront(detectResult);
            if (ret) {
                //执行鼠标操作
                //std::cout << to_string(detectResult.classIds.size());
                //先检查是否设置了自动追踪
                //增加条件，只有使用背包1和2(使用步枪和狙击枪的时候)，才进行追踪，使用其他背包不追踪
                if (m_AssistConfig->autoTrace && (m_weaponInfo.bag==1 || m_weaponInfo.bag == 2)) {
                    //在检查是否已经瞄准了
                    bool isInTarget = mouseKeyboard->IsInTarget(detectResult);
                    //没有瞄准的情况下，才执行鼠标追踪操作
                    if (isInTarget) {
                        //开枪和鼠标移动操作放在不同线程，导致操作割裂，先放回同一个线程处理
                        //增加一个条件，没有人工按下鼠标左键的情况下，才执行自动开枪
                        if (m_AssistConfig->autoFire && !AssistWorker::m_startFire) {

                            mouseKeyboard->AutoFire(detectResult);

                            //由于没有严格的并发控制，自动开火和手动开火有时会冲突
                            //自动开火后做一个补偿，如果检测到手动开火标志，则把鼠标左键再下压
                            if (AssistWorker::m_startFire) {
                                mouseKeyboard->MouseLBDown();
                            }
                        }
                    }
                    else {
                        mouseKeyboard->AutoMove(detectResult);
                    }
                }
            }
        }
    }

    return;
}

void AssistWorker::DrawWork()
{
    //cv::namedWindow("opencvwindows", WINDOW_AUTOSIZE);
    //cv::startWindowThread();

    while (!m_stopFlag)
    {
        if (m_drawPauseFlag)
        {
            //根据线程标志进行线程阻塞
            unique_lock<mutex> locker(m_drawMutex);
            while (m_drawPauseFlag)
            {
                m_drawCondition.wait(locker); // Unlock _mutex and wait to be notified
            }
            locker.unlock();
        }
        else {
            //获取队列中存放的检测结果
            DrawResults drawResult;
            bool ret = drawQueue->PopFront(drawResult);
            if (ret) {
                //执行绘图操作
                DETECTRESULTS out = drawResult.out;
                Mat mat = drawResult.mat;
                double duration = drawResult.duration;
                if (!mat.empty()) {
                    string times = format("%.2f ms", duration);
                    putText(mat, times, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 250), 1);

                    //注意游戏屏幕中心，和检测区域的中心位置不一样，检测区域有向上稍微调整1/10
                    Rect center = { mat.cols/2 -5,mat.rows / 2 + mat.rows /10 - 5,10,10 };
                    rectangle(mat, center, Scalar(0, 0, 250), 2);

                    //画出选定的目标
                    if (out.classIds.size() > 0) {
                        Rect rect = out.boxes[out.maxPersonConfidencePos];
                        rectangle(mat, { rect.x + rect.width/2 -4, rect.y + rect.height / 2 - 4, 8, 8 }, Scalar(255, 178, 50), 2);
                    }

                    for (int i = 0; i < out.classIds.size(); i++) {
                        rectangle(mat, out.boxes[i], Scalar(255, 178, 50), 2);

                        //Get the label for the class name and its confidence
                        string label = format("%.2f", out.confidences[i]);
                        //label = m_classLabels[classIds[i]-1] + ":" + label;
                        //label = "" + to_string(out.classIds[i]) + ", " + label;        
                        label = "" + label + "," + format("%.2f", out.xvals[i]);

                        //Display the label at the top of the bounding box
                        int baseLine;
                        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseLine);
                        int top = max(out.boxes[i].y, labelSize.height);
                        putText(mat, label, Point(out.boxes[i].x, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 250), 1);
                    }
                    
                    //把处理处理好的mat对象放在外部使用的队列中
                    bool push = outDrawQueue->PushBack(mat);
                    //如果队列推送失败，手工清理clone的mat对象
                    if (!push) {
                        mat.release();
                        mat = NULL;
                    }
                   
                    //cv::imshow("opencvwindows", mat);
                    //waitKey(10);
                }
            }
        }
    }

    return;
}

void AssistWorker::DrawAimWork()
{
    //cv::namedWindow("opencvwindows", WINDOW_AUTOSIZE);
    //cv::startWindowThread();

    while (!m_stopFlag)
    {
        if (m_drawAimPauseFlag)
        {
            //根据线程标志进行线程阻塞
            unique_lock<mutex> locker(m_drawAimMutex);
            while (m_drawAimPauseFlag)
            {
                m_drawAimCondition.wait(locker); // Unlock _mutex and wait to be notified
            }
            locker.unlock();
        }
        else {
            //获取队列中存放的检测结果
            if (m_AssistConfig->drawAim) {
                //绘制准星
                drawImage->drawAim();
                Sleep(10);
            }
            else {
                Sleep(500);
            }
        }
    }

    return;
}

void AssistWorker::MouseKeyboardHookWork()
{
    //进入线程时获取windows的线程id
    m_hookThreadId = GetCurrentThreadId();
    while (!m_stopFlag)
    {
        //有些游戏在启动时会清理hook，所以这里通过m_hookPauseFlag标志控制hook可以重新挂载
        if (!m_hookPauseFlag) {
            //挂载hook，哪个线程挂载的hook，就在那个线程中执行回调函数
            if (!m_mouseHook) {
                m_mouseHook = SetWindowsHookEx(WH_MOUSE_LL, MouseHookProcedure, GetModuleHandle(nullptr), NULL);
            }
            if (!m_keyboardHook) {
                m_keyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardHookProcedure, GetModuleHandle(nullptr), NULL);
            }

            //调用GetMessage等相关函数时就自动创建消息队列
            MSG Msg;
            while (!m_stopFlag && GetMessage(&Msg, nullptr, 0, 0) > 0)
            {
                //通过对这个线程消息队列发送WM_AIASSISTPAUSE消息来控制退出消息处理循环
                if (Msg.message == WM_AIASSISTPAUSE) {
                    break;
                }
                TranslateMessage(&Msg);
                DispatchMessage(&Msg);
            }

            if (m_mouseHook) {
                UnhookWindowsHookEx(m_mouseHook);
                m_mouseHook = NULL;
            }
            if (m_keyboardHook) {
                UnhookWindowsHookEx(m_keyboardHook);
                m_keyboardHook = NULL;
            }
        }
        else {
            Sleep(500);
        }
    }
    //退出线程时设置windows的线程id为0
    m_hookThreadId = 0;

    return;
}


void AssistWorker::PushWork()
{
    int randomMax1 = m_AssistConfig->maxJuPushCount;//背包1压枪次数重置阈值，缺省放狙
    int randomMax2 = m_AssistConfig->maxBuPushCount;//背包2压枪次数重置阈值，缺省放步枪

    while (!m_stopFlag)
    {
        if (m_pushPauseFlag || !m_startPush)
        {
            //根据线程标志进行线程阻塞
            unique_lock<mutex> locker(m_pushMutex);
            while (m_pushPauseFlag || !m_startPush)
            {
                m_pushCondition.wait(locker); // Unlock _mutex and wait to be notified
            }
            locker.unlock();

            //每次触发压枪前，再算一次随机数
            randomMax1 = m_AssistConfig->maxJuPushCount;//背包1压枪次数重置阈值，缺省放狙

            //int min = m_AssistConfig->maxBuPushCount, max = m_AssistConfig->maxBuPushCount * 2;
            //randomMax2 = (rand() % (max - min)) + min;//背包2压枪次数重置阈值，缺省放步枪
            randomMax2 = m_AssistConfig->maxBuPushCount;//背包2压枪次数重置阈值，缺省放步枪
        }
        else {

            //如果压枪次数超过阈值，则重置数据左键
             //只对1、2背包重置
            if ((m_weaponInfo.bag == 1 && m_pushCount > randomMax1) || (m_weaponInfo.bag == 2 && m_pushCount > randomMax2)) {
                m_pushCount = 0;
               
                mouseKeyboard->MouseLBUp();
                mouseKeyboard->AutoPush(m_weaponInfo);
                Sleep(150);
                if(m_startPush)
                    mouseKeyboard->MouseLBDown();
            }
            else {
                //执行压枪操作
                mouseKeyboard->AutoPush(m_weaponInfo);
                m_pushCount++;
            }
        }
    }

    return;
}

Mat AssistWorker::PopDrawMat() {
    Mat mat;
    outDrawQueue->PopFront(mat);

    //注意返回克隆对象，才能安全地传递图像数据
    Mat mat2 = mat.clone();

    //释放老mat
    mat.release();
    mat = NULL;

    return mat2;
}

void AssistWorker::Start()
{
    m_detectPauseFlag = false;
    m_detectCondition.notify_all();
    
    m_firePauseFlag = false;
    m_fireCondition.notify_all();

    m_movePauseFlag = false;
    m_moveCondition.notify_all();

    m_drawPauseFlag = false;
    m_drawCondition.notify_all();

    m_drawAimPauseFlag = false;
    m_drawAimCondition.notify_all();

    m_pushPauseFlag = false;
    m_pushCondition.notify_all();

    m_hookPauseFlag = false;

    return;
}

void AssistWorker::Pause()
{
    m_detectPauseFlag = true;
    m_detectCondition.notify_all();

    m_firePauseFlag = true;
    m_fireCondition.notify_all();

    m_movePauseFlag = true;
    m_moveCondition.notify_all();

    m_drawPauseFlag = true;
    m_drawCondition.notify_all();

    m_drawAimPauseFlag = true;
    m_drawAimCondition.notify_all();

    m_pushPauseFlag = true;
    m_pushCondition.notify_all();

    m_hookPauseFlag = true;
    if (mouseKeyboardHookThread != NULL && m_hookThreadId>0) {
        //结hook线程发送退出消息
        PostThreadMessage(m_hookThreadId, WM_AIASSISTPAUSE, 0, 0);
    }
    return;
}
