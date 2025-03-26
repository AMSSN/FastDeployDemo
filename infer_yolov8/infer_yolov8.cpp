#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include<sstream>
#include <time.h>
#include "fastdeploy/vision.h"

void CpuInfer(const std::string& model_file, const std::string& image_file) {
    auto model = fastdeploy::vision::detection::YOLOv8(model_file);
    if (!model.Initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return;
    }

    auto im = cv::imread(image_file);

    fastdeploy::vision::DetectionResult res;
    if (!model.Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return;
    }
    std::cout << res.Str() << std::endl;

    auto vis_im = fastdeploy::vision::VisDetection(im, res);
    cv::imwrite("vis_result.jpg", vis_im);
    std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

void GpuInfer(const std::string& model_file, const std::string& image_file) {
    auto option = fastdeploy::RuntimeOption();
    option.UseGpu();
    auto model = fastdeploy::vision::detection::YOLOv8(model_file, "", option);
    if (!model.Initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return;
    }

    auto im = cv::imread(image_file);

    fastdeploy::vision::DetectionResult res;
    if (!model.Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return;
    }
    std::cout << res.Str() << std::endl;

    auto vis_im = fastdeploy::vision::VisDetection(im, res);
    cv::imwrite("vis_result.jpg", vis_im);
    std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

void TrtInfer(const std::string& model_file, const std::string& image_file) {
    auto option = fastdeploy::RuntimeOption();
    option.UseGpu();
    option.UseTrtBackend();
    option.SetTrtInputShape("images", { 1, 3, 640, 640 });
    auto model = fastdeploy::vision::detection::YOLOv8(model_file, "", option);
    if (!model.Initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return;
    }

    auto im = cv::imread(image_file);

    fastdeploy::vision::DetectionResult res;
    if (!model.Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return;
    }
    std::cout << res.Str() << std::endl;

    auto vis_im = fastdeploy::vision::VisDetection(im, res);
    cv::imwrite("vis_result.jpg", vis_im);
    std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
    cv::VideoCapture cap;   //声明相机捕获对象
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640); //图像的宽，需要相机支持此宽
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640); //图像的高，需要相机支持此高
    //图像分辨率640×640
    int deviceID = 0; //相机设备号
    cap.open(deviceID); //打开相机
    if (!cap.isOpened()) //判断相机是否打开
    {
        std::cerr << "ERROR!!Unable to open camera\n";
        return -1;
    }
    auto model = fastdeploy::vision::detection::YOLOv8("yolov8s.onnx");
    if (!model.Initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return -1;
    }
    fastdeploy::vision::DetectionResult res;

    

    clock_t start, end;
    
    cv::Mat img;
    while (true)
    {
        cap >> img; //以流形式捕获图像
        start = clock();
        model.Predict(img, &res);
        end = clock();   //结束时间
        std::cout << "time = " << double(end - start) << "ms" << std::endl;
        auto vis_im = fastdeploy::vision::VisDetection(img, res);
        //std::cout << res.Str() << std::endl;


        cv::namedWindow("example", 1); //创建一个窗口用于显示图像，1代表窗口适应图像的分辨率进行拉伸。
        if (img.empty() == false) //图像不为空则显示图像
        {
            cv::imshow("example", vis_im);
        }

        int  key = cv::waitKey(30); //等待30ms
        if (key == int('q')) //按下q退出
        {
            break;
        }

    }
    cap.release(); //释放相机捕获对象
    cv::destroyAllWindows(); //关闭所有窗口
    return 0;
}