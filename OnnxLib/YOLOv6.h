
#ifndef LITE_AI_TOOLKIT_ORT_CV_YOLOV6_H
#define LITE_AI_TOOLKIT_ORT_CV_YOLOV6_H

#include "BasicOrtHandler.h"
#include "OrtUtils.h"
#include "LiteUtils.h"

namespace onnxlib
{
    class YOLOv6 : public BasicOrtHandler
    {
    public:
        explicit YOLOv6(const std::string& _onnx_path, unsigned int _num_threads = 1) :
            BasicOrtHandler(_onnx_path, _num_threads)
        {};

        ~YOLOv6() override = default;

    private:
        // nested classes
        typedef struct
        {
            float r;
            int dw;
            int dh;
            int new_unpad_w;
            int new_unpad_h;
            bool flag;
        } YOLOv6ScaleParams;

    private:
        const char* class_names[80] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        };
        enum NMS
        {
            HARD = 0, BLEND = 1, OFFSET = 2
        };
        static constexpr const unsigned int max_nms = 30000;

    private:
        Ort::Value transform(const cv::Mat& mat_rs) override; // without resize

        void resize_unscale(const cv::Mat& mat,
            cv::Mat& mat_rs,
            int target_height,
            int target_width,
            YOLOv6ScaleParams& scale_params);

        void generate_bboxes(const YOLOv6ScaleParams& scale_params,
            std::vector<lite::types::Boxf>& bbox_collection,
            std::vector<Ort::Value>& output_tensors,
            float score_threshold, int img_height,
            int img_width); // rescale & exclude

        void nms(std::vector<lite::types::Boxf>& input, std::vector<lite::types::Boxf>& output,
            float iou_threshold, unsigned int topk, unsigned int nms_type);

    public:
        void detect(const cv::Mat& mat, std::vector<lite::types::Boxf>& detected_boxes,
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

    };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_YOLOV6_H

