
**游戏助手已发布三个项目，区别如下：**<br>
AIAssistOnnx 使用onnx进行AI推理，onnx在终端上运行时性能很高，实测比opencv dnn快2-3倍 <br>
这个版本使用onnx+yoyov6，游戏图像检测速度快到飞起。。。<br>

**已发布可运行程序：包含了相关运行库文件，可以到到项目Releases中下载AIAssist.rar，解压就可以直接运行。**
<br>

**重构内容：** <br>
1、替换AI推理依赖的库，使用onnx替换opencv dnn；<br>


**可以考虑的推理加速改进方向：** <br>
1、使用CPU推理，配置多核CPU，能上两块物理CPU更佳，可以使用一块CPU专用于推理；<br>
2、使用GPU推理，onnx提供预编译的gpu版本，可以平替cpu版本，使用方便；<br>
<br>


**一、项目说明：**<br>
AIAssistC是一个AI游戏助手，使用OpenCv、onnx、yoyo、MFC等技术，截取游戏屏幕，使用AI模型进行对象识别，并实现自动瞄准/自动开枪等鼠标操作，提升玩家的游戏体验。<br>
<br>

**二、工程说明：**<br>
AIAssist：mfc前端UI子工程；<br>
AIAssistLib：AI助手静态库子工程；<br>
OnnxLib：onnx推理封装静态库子工程；<br>
OnnxTest：onnx推理测试子工程；<br>
Data：存放模型文件及工具文件的子工程；<br>
DXGICaptureSample：DXGI Desktop Duplication API截屏测试子工程；<br>
OpencvTest：openc功能验证测试子工程；<br>
<br>

**三、主要的运行库：**<br>
1、intel贡献的大神级图像处理框架OpenCv：<br>
https://opencv.org/ <br>

2、微软onnx AI运行库，目前主流AI框架大都支持把模型转换为onnx格式：<br>
https://onnxruntime.ai/ <br>
https://github.com/DefTruth/lite.ai.toolkit  onnx c++样例代码比较难找，我主要参考了这个项目，作者封装了大量模型调用，NB加感谢<br>

3、模型使用美团的yoyov6，准确率和性能就一个字，NB： <br>
https://github.com/meituan/YOLOv6 <br>
<br>

**四、使用注意：**<br>
1、使用技巧：目前基于AI图像检测，只做到了人员识别，无法区分敌我。为避免游戏中自动追踪并射击队友的尴尬，参考如下使用技巧。习惯这些操作方式后，这个工具使用体验相对好一些。:<br>
a、按其他数字键停止自动追踪：目前在工具中加入了控制逻辑，只有切换到主副武器时才会运行自动追踪和自动射击，切换到其他武器时不会自动追踪。
所以在捡枪或者跟队友跑时，切换到手枪或其他数字键，停止自动追踪。对敌时再切换到主副武器（数字键1或2），自动追踪才生效。<br>
b、避免追踪游戏者自己：目前无法区分识别出来的图像是自己还是其他人，钢枪或者搜索时切换为T第一人称视图，避免检测到游戏者自己。<br>

2、使用windows hook及鼠标键盘api实现了鼠标键盘操作模拟，HIDDriver驱动程序不再是必须项。在绝地求生、逆战、穿越火线三个游戏上测试，可以正常工作。推测现在的游戏安全检测侧重游戏内部的行为数据检测，游戏外部环境中windwos hook等通用操作机制不再管控<br>

3、HIDDriver鼠标键盘模拟驱动已移除，这个驱动没有微软颁发的正式证书，只能在win 10测试模式下执行。如果要尝试，参考项目说明： <br>
https://github.com/dengqizhou30/HIDDriver <br>
<br>

**五、游戏截图：**<br>
工具刚改完，正在体验中，还没来得及截图。。。<br>
