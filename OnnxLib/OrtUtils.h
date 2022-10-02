
#ifndef AI_LIB_ORT_UTILS_H
#define AI_LIB_ORT_UTILS_H

#include <onnxruntime_cxx_api.h>
#include <opencv.hpp>

namespace onnxlib
{
    enum
    {
        CHW = 0, HWC = 1
    };

    /**
        * @param mat CV:Mat with type 'CV_32FC3|2|1'
        * @param tensor_dims e.g {1,C,H,W} | {1,H,W,C}
        * @param memory_info It needs to be a global variable in a class
        * @param tensor_value_handler It needs to be a global variable in a class
        * @param data_format CHW | HWC
        * @return
        */
    Ort::Value create_tensor(const cv::Mat& mat, const std::vector<int64_t>& tensor_dims,
        const Ort::MemoryInfo& memory_info_handler,
        std::vector<float>& tensor_value_handler,
        unsigned int data_format = CHW) throw(std::runtime_error);

    cv::Mat normalize(const cv::Mat& mat, float mean, float scale);

    cv::Mat normalize(const cv::Mat& mat, const float mean[3], const float scale[3]);

    void normalize(const cv::Mat& inmat, cv::Mat& outmat, float mean, float scale);

    void normalize_inplace(cv::Mat& mat_inplace, float mean, float scale);

    void normalize_inplace(cv::Mat& mat_inplace, const float mean[3], const float scale[3]);


   
} // NAMESPACE ORTCV

#endif //AI_LIB_ORT_UTILS_H