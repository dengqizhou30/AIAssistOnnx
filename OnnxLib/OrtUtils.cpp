#include "pch.h"
#include "OrtUtils.h"
#include <comdef.h>

using namespace std;

//*************************************** ortcv::utils **********************************************//
Ort::Value onnxlib::create_tensor(const cv::Mat& mat,
    const std::vector<int64_t>& tensor_dims,
    const Ort::MemoryInfo& memory_info_handler,
    std::vector<float>& tensor_value_handler,
    unsigned int data_format)
    throw(std::runtime_error)
{
    const unsigned int rows = mat.rows;
    const unsigned int cols = mat.cols;
    const unsigned int channels = mat.channels();

    cv::Mat mat_ref;
    if (mat.type() != CV_32FC(channels)) mat.convertTo(mat_ref, CV_32FC(channels));
    else mat_ref = mat; // reference only. zero-time cost. support 1/2/3/... channels

    if (tensor_dims.size() != 4) throw std::runtime_error("dims mismatch.");
    if (tensor_dims.at(0) != 1) throw std::runtime_error("batch != 1");

    // CXHXW
    if (data_format == onnxlib::CHW)
    {

        const unsigned int target_height = tensor_dims.at(2);
        const unsigned int target_width = tensor_dims.at(3);
        const unsigned int target_channel = tensor_dims.at(1);
        const unsigned int target_tensor_size = target_channel * target_height * target_width;
        if (target_channel != channels) throw std::runtime_error("channel mismatch.");

        tensor_value_handler.resize(target_tensor_size);

        cv::Mat resize_mat_ref;
        if (target_height != rows || target_width != cols)
            cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
        else resize_mat_ref = mat_ref; // reference only. zero-time cost.

        std::vector<cv::Mat> mat_channels;
        cv::split(resize_mat_ref, mat_channels);
        // CXHXW
        for (unsigned int i = 0; i < channels; ++i)
            std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
                mat_channels.at(i).data, target_height * target_width * sizeof(float));

        return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
            target_tensor_size, tensor_dims.data(),
            tensor_dims.size());
    }

    // HXWXC
    const unsigned int target_height = tensor_dims.at(1);
    const unsigned int target_width = tensor_dims.at(2);
    const unsigned int target_channel = tensor_dims.at(3);
    const unsigned int target_tensor_size = target_channel * target_height * target_width;
    if (target_channel != channels) throw std::runtime_error("channel mismatch!");
    tensor_value_handler.resize(target_tensor_size);

    cv::Mat resize_mat_ref;
    if (target_height != rows || target_width != cols)
        cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
    else resize_mat_ref = mat_ref; // reference only. zero-time cost.

    std::memcpy(tensor_value_handler.data(), resize_mat_ref.data, target_tensor_size * sizeof(float));

    return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
        target_tensor_size, tensor_dims.data(),
        tensor_dims.size());
}

cv::Mat onnxlib::normalize(const cv::Mat& mat, float mean, float scale)
{
    cv::Mat matf;
    if (mat.type() != CV_32FC3) mat.convertTo(matf, CV_32FC3);
    else matf = mat; // reference
    return (matf - mean) * scale;
}

cv::Mat onnxlib::normalize(const cv::Mat& mat, const float* mean, const float* scale)
{
    cv::Mat mat_copy;
    if (mat.type() != CV_32FC3) mat.convertTo(mat_copy, CV_32FC3);
    else mat_copy = mat.clone();
    for (unsigned int i = 0; i < mat_copy.rows; ++i)
    {
        cv::Vec3f* p = mat_copy.ptr<cv::Vec3f>(i);
        for (unsigned int j = 0; j < mat_copy.cols; ++j)
        {
            p[j][0] = (p[j][0] - mean[0]) * scale[0];
            p[j][1] = (p[j][1] - mean[1]) * scale[1];
            p[j][2] = (p[j][2] - mean[2]) * scale[2];
        }
    }
    return mat_copy;
}

void onnxlib::normalize(const cv::Mat& inmat, cv::Mat& outmat,
    float mean, float scale)
{
    outmat = onnxlib::normalize(inmat, mean, scale);
}

void onnxlib::normalize_inplace(cv::Mat& mat_inplace, float mean, float scale)
{
    if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
    onnxlib::normalize(mat_inplace, mat_inplace, mean, scale);
}

void onnxlib::normalize_inplace(cv::Mat& mat_inplace, const float* mean, const float* scale)
{
    if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
    for (unsigned int i = 0; i < mat_inplace.rows; ++i)
    {
        cv::Vec3f* p = mat_inplace.ptr<cv::Vec3f>(i);
        for (unsigned int j = 0; j < mat_inplace.cols; ++j)
        {
            p[j][0] = (p[j][0] - mean[0]) * scale[0];
            p[j][1] = (p[j][1] - mean[1]) * scale[1];
            p[j][2] = (p[j][2] - mean[2]) * scale[2];
        }
    }
}



//*************************************** ortcv::utils **********************************************//








