#include "pch.h"
#include "LiteUtils.h"

//*************************************** lite::utils **********************************************//
// String Utils
std::string lite::utils::to_string(const std::wstring& wstr)
{
    unsigned len = wstr.size() * 4;
    setlocale(LC_CTYPE, "");
    char* p = new char[len];
    //wcstombs(p, wstr.c_str(), len);
    size_t conv;
    wcstombs_s(&conv, p, (size_t)len, wstr.c_str(), (size_t)len-1);
    std::string str(p);
    delete[] p;
    return str;
}

std::wstring lite::utils::to_wstring(const std::string& str)
{
    unsigned len = str.size() * 2;
    setlocale(LC_CTYPE, "");
    wchar_t* p = new wchar_t[len];
    //mbstowcs(p, str.c_str(), len);
    size_t conv;
    mbstowcs_s(&conv, p, (size_t)len, str.c_str(), (size_t)len - 1);
    std::wstring wstr(p);
    delete[] p;
    return wstr;
}

// Drawing Utils
// reference: https://github.com/DefTruth/headpose-fsanet-pytorch/blob/master/src/utils.py
void lite::utils::draw_axis_inplace(cv::Mat& mat_inplace,
    const types::EulerAngles& euler_angles,
    float size, int thickness)
{
    if (!euler_angles.flag) return;

    const float pitch = euler_angles.pitch * _PI / 180.f;
    const float yaw = -euler_angles.yaw * _PI / 180.f;
    const float roll = euler_angles.roll * _PI / 180.f;

    const float height = static_cast<float>(mat_inplace.rows);
    const float width = static_cast<float>(mat_inplace.cols);

    const int tdx = static_cast<int>(width / 2.0f);
    const int tdy = static_cast<int>(height / 2.0f);

    // X-Axis pointing to right. drawn in red
    const int x1 = static_cast<int>(size * std::cos(yaw) * std::cos(roll)) + tdx;
    const int y1 = static_cast<int>(
        size * (std::cos(pitch) * std::sin(roll)
            + std::cos(roll) * std::sin(pitch) * std::sin(yaw))
        ) + tdy;
    // Y-Axis | drawn in green
    const int x2 = static_cast<int>(-size * std::cos(yaw) * std::sin(roll)) + tdx;
    const int y2 = static_cast<int>(
        size * (std::cos(pitch) * std::cos(roll)
            - std::sin(pitch) * std::sin(yaw) * std::sin(roll))
        ) + tdy;
    // Z-Axis (out of the screen) drawn in blue
    const int x3 = static_cast<int>(size * std::sin(yaw)) + tdx;
    const int y3 = static_cast<int>(-size * std::cos(yaw) * std::sin(pitch)) + tdy;

    cv::line(mat_inplace, cv::Point2i(tdx, tdy), cv::Point2i(x1, y1), cv::Scalar(0, 0, 255), thickness);
    cv::line(mat_inplace, cv::Point2i(tdx, tdy), cv::Point2i(x2, y2), cv::Scalar(0, 255, 0), thickness);
    cv::line(mat_inplace, cv::Point2i(tdx, tdy), cv::Point2i(x3, y3), cv::Scalar(255, 0, 0), thickness);
}

cv::Mat lite::utils::draw_axis(const cv::Mat& mat,
    const types::EulerAngles& euler_angles,
    float size, int thickness)
{
    if (!euler_angles.flag) return mat;

    cv::Mat mat_copy = mat.clone();
    const float pitch = euler_angles.pitch * _PI / 180.f;
    const float yaw = -euler_angles.yaw * _PI / 180.f;
    const float roll = euler_angles.roll * _PI / 180.f;

    const float height = static_cast<float>(mat_copy.rows);
    const float width = static_cast<float>(mat_copy.cols);

    const int tdx = static_cast<int>(width / 2.0f);
    const int tdy = static_cast<int>(height / 2.0f);

    // X-Axis pointing to right. drawn in red
    const int x1 = static_cast<int>(size * std::cos(yaw) * std::cos(roll)) + tdx;
    const int y1 = static_cast<int>(
        size * (std::cos(pitch) * std::sin(roll)
            + std::cos(roll) * std::sin(pitch) * std::sin(yaw))
        ) + tdy;
    // Y-Axis | drawn in green
    const int x2 = static_cast<int>(-size * std::cos(yaw) * std::sin(roll)) + tdx;
    const int y2 = static_cast<int>(
        size * (std::cos(pitch) * std::cos(roll)
            - std::sin(pitch) * std::sin(yaw) * std::sin(roll))
        ) + tdy;
    // Z-Axis (out of the screen) drawn in blue
    const int x3 = static_cast<int>(size * std::sin(yaw)) + tdx;
    const int y3 = static_cast<int>(-size * std::cos(yaw) * std::sin(pitch)) + tdy;

    cv::line(mat_copy, cv::Point2i(tdx, tdy), cv::Point2i(x1, y1), cv::Scalar(0, 0, 255), thickness);
    cv::line(mat_copy, cv::Point2i(tdx, tdy), cv::Point2i(x2, y2), cv::Scalar(0, 255, 0), thickness);
    cv::line(mat_copy, cv::Point2i(tdx, tdy), cv::Point2i(x3, y3), cv::Scalar(255, 0, 0), thickness);

    return mat_copy;
}

cv::Mat lite::utils::draw_landmarks(const cv::Mat& mat, types::Landmarks& landmarks)
{
    if (landmarks.points.empty() || !landmarks.flag) return mat;
    cv::Mat mat_copy = mat.clone();
    for (const auto& p : landmarks.points)
        cv::circle(mat_copy, p, 2, cv::Scalar(0, 255, 0), -1);
    return mat_copy;
}

void lite::utils::draw_landmarks_inplace(cv::Mat& mat, types::Landmarks& landmarks)
{
    if (landmarks.points.empty() || !landmarks.flag) return;
    for (const auto& p : landmarks.points)
        cv::circle(mat, p, 2, cv::Scalar(0, 255, 0), -1);
}

void lite::utils::draw_boxes_inplace(cv::Mat& mat_inplace, const std::vector<types::Boxf>& boxes)
{
    if (boxes.empty()) return;
    int count = 0;
    for (const auto& box : boxes)
    {
        if (box.flag)
        {
            cv::rectangle(mat_inplace, box.rect(), cv::Scalar(255, 0, 0), 1);
            //第一个最大置信度的方框中间画一个标志
            if (count == 0) {
                cv::Rect rect = box.rect();
                cv::Rect crect;
                crect.x = rect.x + rect.width / 2 - 5;
                crect.y = rect.y + rect.height / 2 - 5;
                crect.width = 10;
                crect.height = 10;
                cv::rectangle(mat_inplace, crect, cv::Scalar(0, 0, 255), 1);
            }
            if (box.label_text)
            {
                std::string label_text(box.label_text);
                label_text = label_text + ":" + std::to_string(box.score).substr(0, 4);
                cv::putText(mat_inplace, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    0.6f, cv::Scalar(0, 255, 0), 1);

            }

            count++;
        }
    }
}

void lite::utils::draw_boxes_with_landmarks_inplace(cv::Mat& mat_inplace, const std::vector<types::BoxfWithLandmarks>& boxes_kps, bool text)
{
    if (boxes_kps.empty()) return;
    for (const auto& box_kps : boxes_kps)
    {
        if (box_kps.flag)
        {
            // box
            if (box_kps.box.flag)
            {
                cv::rectangle(mat_inplace, box_kps.box.rect(), cv::Scalar(255, 255, 0), 2);
                if (box_kps.box.label_text && text)
                {
                    std::string label_text(box_kps.box.label_text);
                    label_text = label_text + ":" + std::to_string(box_kps.box.score).substr(0, 4);
                    cv::putText(mat_inplace, label_text, box_kps.box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                        0.6f, cv::Scalar(0, 255, 0), 2);

                }
            }
            // landmarks
            if (box_kps.landmarks.flag && !box_kps.landmarks.points.empty())
            {
                for (const auto& point : box_kps.landmarks.points)
                    cv::circle(mat_inplace, point, 2, cv::Scalar(0, 255, 0), -1);
            }
        }
    }
}

cv::Mat lite::utils::draw_boxes_with_landmarks(const cv::Mat& mat, const std::vector<types::BoxfWithLandmarks>& boxes_kps, bool text)
{
    if (boxes_kps.empty()) return mat;
    cv::Mat canva = mat.clone();
    for (const auto& box_kps : boxes_kps)
    {
        if (box_kps.flag)
        {
            // box
            if (box_kps.box.flag)
            {
                cv::rectangle(canva, box_kps.box.rect(), cv::Scalar(255, 255, 0), 2);
                if (box_kps.box.label_text && text)
                {
                    std::string label_text(box_kps.box.label_text);
                    label_text = label_text + ":" + std::to_string(box_kps.box.score).substr(0, 4);
                    cv::putText(canva, label_text, box_kps.box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                        0.6f, cv::Scalar(0, 255, 0), 2);

                }
            }
            // landmarks
            if (box_kps.landmarks.flag && !box_kps.landmarks.points.empty())
            {
                for (const auto& point : box_kps.landmarks.points)
                    cv::circle(canva, point, 2, cv::Scalar(0, 255, 0), -1);
            }
        }
    }
    return canva;
}

cv::Mat lite::utils::draw_age(const cv::Mat& mat, types::Age& age)
{
    if (!age.flag) return mat;
    cv::Mat canva = mat.clone();
    const unsigned int offset = static_cast<unsigned int>(
        0.1f * static_cast<float>(mat.rows));
    std::string age_text = "Age:" + std::to_string(age.age).substr(0, 4);
    std::string interval_text = "Interval" + std::to_string(age.age_interval[0])
        + "_" + std::to_string(age.age_interval[1]);
    std::string interval_prob = "Prob:" + std::to_string(age.interval_prob).substr(0, 4);
    cv::putText(canva, age_text, cv::Point2i(10, offset),
        cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
    cv::putText(canva, interval_text, cv::Point2i(10, offset * 2),
        cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(255, 0, 0), 2);
    cv::putText(canva, interval_prob, cv::Point2i(10, offset * 3),
        cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 0, 255), 2);
    return canva;
}

void lite::utils::draw_age_inplace(cv::Mat& mat_inplace, types::Age& age)
{
    if (!age.flag) return;
    const unsigned int offset = static_cast<unsigned int>(
        0.1f * static_cast<float>(mat_inplace.rows));
    std::string age_text = "Age:" + std::to_string(age.age).substr(0, 4);
    std::string interval_text = "Interval" + std::to_string(age.age_interval[0])
        + "_" + std::to_string(age.age_interval[1]);
    std::string interval_prob = "Prob:" + std::to_string(age.interval_prob).substr(0, 4);
    cv::putText(mat_inplace, age_text, cv::Point2i(10, offset),
        cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
    cv::putText(mat_inplace, interval_text, cv::Point2i(10, offset * 2),
        cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(255, 0, 0), 2);
    cv::putText(mat_inplace, interval_prob, cv::Point2i(10, offset * 3),
        cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 0, 255), 2);
}

cv::Mat lite::utils::draw_gender(const cv::Mat& mat, types::Gender& gender)
{
    if (!gender.flag) return mat;
    cv::Mat canva = mat.clone();
    const unsigned int offset = static_cast<unsigned int>(
        0.1f * static_cast<float>(mat.rows));
    std::string gender_text = std::to_string(gender.label) + ":"
        + gender.text + ":" + std::to_string(gender.score).substr(0, 4);
    cv::putText(canva, gender_text, cv::Point2i(10, offset),
        cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
    return canva;
}

void lite::utils::draw_gender_inplace(cv::Mat& mat_inplace, types::Gender& gender)
{
    if (!gender.flag) return;
    const unsigned int offset = static_cast<unsigned int>(
        0.1f * static_cast<float>(mat_inplace.rows));
    std::string gender_text = std::to_string(gender.label) + ":"
        + gender.text + ":" + std::to_string(gender.score).substr(0, 4);
    cv::putText(mat_inplace, gender_text, cv::Point2i(10, offset),
        cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
}

cv::Mat lite::utils::draw_emotion(const cv::Mat& mat, types::Emotions& emotions)
{
    if (!emotions.flag) return mat;
    cv::Mat canva = mat.clone();
    const unsigned int offset = static_cast<unsigned int>(
        0.1f * static_cast<float>(mat.rows));
    std::string emotion_text = std::to_string(emotions.label) + ":"
        + emotions.text + ":" + std::to_string(emotions.score).substr(0, 4);
    cv::putText(canva, emotion_text, cv::Point2i(10, offset),
        cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
    return canva;
}

void lite::utils::draw_emotion_inplace(cv::Mat& mat_inplace, types::Emotions& emotions)
{
    if (!emotions.flag) return;
    const unsigned int offset = static_cast<unsigned int>(
        0.1f * static_cast<float>(mat_inplace.rows));
    std::string emotion_text = std::to_string(emotions.label) + ":"
        + emotions.text + ":" + std::to_string(emotions.score).substr(0, 4);
    cv::putText(mat_inplace, emotion_text, cv::Point2i(10, offset),
        cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
}

// Object Detection Utils
// reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/
//            blob/master/ncnn/src/UltraFace.cpp
void lite::utils::hard_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
    float iou_threshold, unsigned int topk)
{
    if (input.empty()) return;
    std::sort(input.begin(), input.end(),
        [](const types::Boxf& a, const types::Boxf& b)
        { return a.score > b.score; });
    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i)
    {
        if (merged[i]) continue;
        std::vector<types::Boxf> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j)
        {
            if (merged[j]) continue;

            float iou = static_cast<float>(input[i].iou_of(input[j]));

            if (iou > iou_threshold)
            {
                merged[j] = 1;
                buf.push_back(input[j]);
            }

        }
        output.push_back(buf[0]);

        // keep top k
        count += 1;
        if (count >= topk)
            break;
    }
}

void lite::utils::blending_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
    float iou_threshold, unsigned int topk)
{
    if (input.empty()) return;
    std::sort(input.begin(), input.end(),
        [](const types::Boxf& a, const types::Boxf& b)
        { return a.score > b.score; });
    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i)
    {
        if (merged[i]) continue;
        std::vector<types::Boxf> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j)
        {
            if (merged[j]) continue;

            float iou = static_cast<float>(input[i].iou_of(input[j]));
            if (iou > iou_threshold)
            {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }

        float total = 0.f;
        for (unsigned int k = 0; k < buf.size(); ++k)
        {
            total += std::exp(buf[k].score);
        }
        types::Boxf rects;
        for (unsigned int l = 0; l < buf.size(); ++l)
        {
            float rate = std::exp(buf[l].score) / total;
            rects.x1 += buf[l].x1 * rate;
            rects.y1 += buf[l].y1 * rate;
            rects.x2 += buf[l].x2 * rate;
            rects.y2 += buf[l].y2 * rate;
            rects.score += buf[l].score * rate;
        }
        rects.flag = true;
        output.push_back(rects);

        // keep top k
        count += 1;
        if (count >= topk)
            break;
    }
}

// reference: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
void lite::utils::offset_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
    float iou_threshold, unsigned int topk)
{
    if (input.empty()) return;
    std::sort(input.begin(), input.end(),
        [](const types::Boxf& a, const types::Boxf& b)
        { return a.score > b.score; });
    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    const float offset = 4096.f;
    /** Add offset according to classes.
     * That is, separate the boxes into categories, and each category performs its
     * own NMS operation. The same offset will be used for those predicted to be of
     * the same category. Therefore, the relative positions of boxes of the same
     * category will remain unchanged. Box of different classes will be farther away
     * after offset, because offsets are different. In this way, some overlapping but
     * different categories of entities are not filtered out by the NMS. Very clever!
     */
    for (unsigned int i = 0; i < box_num; ++i)
    {
        input[i].x1 += static_cast<float>(input[i].label) * offset;
        input[i].y1 += static_cast<float>(input[i].label) * offset;
        input[i].x2 += static_cast<float>(input[i].label) * offset;
        input[i].y2 += static_cast<float>(input[i].label) * offset;
    }

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i)
    {
        if (merged[i]) continue;
        std::vector<types::Boxf> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j)
        {
            if (merged[j]) continue;

            float iou = static_cast<float>(input[i].iou_of(input[j]));

            if (iou > iou_threshold)
            {
                merged[j] = 1;
                buf.push_back(input[j]);
            }

        }
        output.push_back(buf[0]);

        // keep top k
        count += 1;
        if (count >= topk)
            break;
    }

    /** Substract offset.*/
    if (!output.empty())
    {
        for (unsigned int i = 0; i < output.size(); ++i)
        {
            output[i].x1 -= static_cast<float>(output[i].label) * offset;
            output[i].y1 -= static_cast<float>(output[i].label) * offset;
            output[i].x2 -= static_cast<float>(output[i].label) * offset;
            output[i].y2 -= static_cast<float>(output[i].label) * offset;
        }
    }

}

// Matting Utils & Segmentation Utils
void lite::utils::swap_background(const cv::Mat& fgr_mat, const cv::Mat& pha_mat,
    const cv::Mat& bgr_mat, cv::Mat& out_mat,
    bool fgr_is_already_mul_pha)
{
    // user-friendly method for background swap.
    if (fgr_mat.empty() || pha_mat.empty() || bgr_mat.empty()) return;
    const unsigned int fg_h = fgr_mat.rows;
    const unsigned int fg_w = fgr_mat.cols;
    const unsigned int bg_h = bgr_mat.rows;
    const unsigned int bg_w = bgr_mat.cols;
    const unsigned int ph_h = pha_mat.rows;
    const unsigned int ph_w = pha_mat.cols;
    const unsigned int channels = fgr_mat.channels();
    if (channels != 3) return; // only support 3 channels.
    const unsigned int num_elements = fg_h * fg_w * channels;

    cv::Mat bg_mat_copy, ph_mat_copy, fg_mat_copy;
    if (bg_h != fg_h || bg_w != fg_w)
        cv::resize(bgr_mat, bg_mat_copy, cv::Size(fg_w, fg_h));
    else bg_mat_copy = bgr_mat; // ref only.
    if (ph_h != fg_h || ph_w != fg_w)
        cv::resize(pha_mat, ph_mat_copy, cv::Size(fg_w, fg_h));
    else ph_mat_copy = pha_mat; // ref only.
    if (ph_mat_copy.channels() == 1)
        cv::cvtColor(ph_mat_copy, ph_mat_copy, cv::COLOR_GRAY2BGR); // 0.~1.
      // convert mats to float32 points.
    if (bg_mat_copy.type() != CV_32FC3) bg_mat_copy.convertTo(bg_mat_copy, CV_32FC3); // 0.~255.
    if (ph_mat_copy.type() != CV_32FC3) ph_mat_copy.convertTo(ph_mat_copy, CV_32FC3); // 0.~1.
    if (fgr_mat.type() != CV_32FC3) fgr_mat.convertTo(fg_mat_copy, CV_32FC3); // 0.~255.
    else fg_mat_copy = fgr_mat; // ref only

    // element wise operations.
    out_mat = fg_mat_copy.clone();
    const float* fg_ptr = (float*)fg_mat_copy.data;
    const float* bg_ptr = (float*)bg_mat_copy.data;
    const float* ph_ptr = (float*)ph_mat_copy.data;
    float* mutable_out_ptr = (float*)out_mat.data;

    // TODO: add omp support instead of native loop.
    if (!fgr_is_already_mul_pha)
        for (unsigned int i = 0; i < num_elements; ++i)
            mutable_out_ptr[i] = fg_ptr[i] * ph_ptr[i] + (1.f - ph_ptr[i]) * bg_ptr[i];
    else
        for (unsigned int i = 0; i < num_elements; ++i)
            mutable_out_ptr[i] = fg_ptr[i] + (1.f - ph_ptr[i]) * bg_ptr[i];

    if (!out_mat.empty() && out_mat.type() != CV_8UC3)
        out_mat.convertTo(out_mat, CV_8UC3);
}

void lite::utils::remove_small_connected_area(cv::Mat& alpha_pred, float threshold)
{
    cv::Mat gray, binary;
    alpha_pred.convertTo(gray, CV_8UC1, 255.f);
    // 255 * 0.05 ~ 13
    unsigned int binary_threshold = (unsigned int)(255.f * threshold);
    // https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/util.py#L209
    cv::threshold(gray, binary, binary_threshold, 255, cv::THRESH_BINARY);
    // morphologyEx with OPEN operation to remove noise first.
    auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    // Computationally connected domain
    cv::Mat labels = cv::Mat::zeros(alpha_pred.size(), CV_32S);
    cv::Mat stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
    if (num_labels <= 1) return; // no noise, skip.
    // find max connected area, 0 is background
    int max_connected_id = 1; // 1,2,...
    int max_connected_area = stats.at<int>(max_connected_id, cv::CC_STAT_AREA);
    for (int i = 1; i < num_labels; ++i)
    {
        int tmp_connected_area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (tmp_connected_area > max_connected_area)
        {
            max_connected_area = tmp_connected_area;
            max_connected_id = i;
        }
    }
    const int h = alpha_pred.rows;
    const int w = alpha_pred.cols;
    // remove small connected area.
    for (int i = 0; i < h; ++i)
    {
        int* label_row_ptr = labels.ptr<int>(i);
        float* alpha_row_ptr = alpha_pred.ptr<float>(i);
        for (int j = 0; j < w; ++j)
        {
            if (label_row_ptr[j] != max_connected_id)
                alpha_row_ptr[j] = 0.f;
        }
    }
}

//*************************************** lite::utils **********************************************//