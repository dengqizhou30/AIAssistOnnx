#include "pch.h"
#include "AssistState.h"


//初始化静态成员变量
AssistConfig* AssistState::m_AssistConfig = AssistConfig::GetInstance();

string AssistState::getStatInf() {
	char info[1024];

	memset(info, 0, sizeof(info));

	string fmt("游戏：%s；\r\n游戏窗口：%d,%d,%d,%d；\r\n检测区域：%d,%d,%d,%d；\r\n自动追踪：%s；\r\n自动开火：%s；\r\n注意：v键及鼠标中键可以关闭或恢复自动追踪/开火/压枪；\r\n开关错乱时按enter键关闭所有操作；\r\n建议：每次交火前按v键切换视图；");

	snprintf(info, sizeof(info), fmt.c_str(), m_AssistConfig->gameName,
		m_AssistConfig->screenRect.x, m_AssistConfig->screenRect.y, m_AssistConfig->screenRect.width, m_AssistConfig->screenRect.height,
		m_AssistConfig->detectRect.x, m_AssistConfig->detectRect.y, m_AssistConfig->detectRect.width, m_AssistConfig->detectRect.height, 
		m_AssistConfig->autoTrace ? "true" : "false", m_AssistConfig->autoFire ? "true" : "false", 1024);

	string ret(info);

	return ret;
}