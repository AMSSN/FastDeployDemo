#### fastdeploy总结

1、用fastdeploy跑通了C++/python版本的yolov8s，后面训练可以用ultralytics的官方代码，然后用onnx模型可以直接部署。

2、OCR的可以用paddle OCR进行训练，fastdeploy部署用pdmodel和pdiparams文件。

3、分类模型可以用paddlecls进行训练，fastdeploy部署用pdmodel和pdiparams文件。

4、

#### fastdeploy安装

下载预编译文件https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md

#### fastdeploy官方支持的模型库

| 模型                | 简介                                                         | 完成情况 | 速度                                                         |
| ------------------- | ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| 分类                |                                                              |          |                                                              |
| pp-LCNet            |                                                              | ✔        | PPLCNet_x1_0(i3-13100CPU)+Paddle Inference推理，一张图片169ms/164ms<br/>**PPLCNet_x1_0(i3-13100CPU)+OenVINO推理，一张图片104ms/107ms**<br/>PPLCNet_x1_0(i3-13100CPU)+ONNX Runtime推理，一张图片173ms/172ms<br />PPLCNet(2080tiGPU)+Paddle Inference推理，一张图片1462ms/1448ms<br/>PPLCNet(2080tiGPU)+Paddle TensorRT推理，一张图片600ms/586ms<br/>PPLCNet(2080tiGPU)+ONNX Runtime推理，一张图片890ms/916ms<br/>PPLCNet(2080tiGPU)+Nvidia TensorRT推理，一张图片74s(不用cache)  2ms(cache) |
| EfficientNet        |                                                              | ✔        | EfficientNetB0(i3-13100CPU)+Paddle Inference推理，一张图片492ms/482ms<br/>**EfficientNetB0(i3-13100CPU)+OenVINO推理，一张图片330ms/205ms**<br/>EfficientNetB0(i3-13100CPU)+ONNX Runtime推理，一张图片372ms/352ms<br/>EfficientNetB7(i3-13100CPU)+Paddle Inference推理，一张图片5701ms/5448ms<br/>**EfficientNetB7(i3-13100CPU)+OenVINO推理，一张图片1190ms/1209ms**<br/>EfficientNetB7(i3-13100CPU)+ONNX Runtime推理，一张图片3942ms/3928ms<br />EfficientNetB0(2080tiGPU)+Paddle TensorRT推理，一张图片600ms/594ms<br/>EfficientNetB0(2080tiGPU)+ONNX Runtime推理，一张图片1459ms/1455ms<br/>EfficientNetB0(2080tiGPU)+Nvidia TensorRT推理，一张图片96s(不用cache)，570ms(cache)<br />EfficientNetB7(2080tiGPU)+Paddle TensorRT推理，一张图片858ms/857ms<br/>EfficientNetB7(2080tiGPU)+ONNX Runtime推理，一张图片8361ms/10266ms<br/>EfficientNetB7(2080tiGPU)+Nvidia TensorRT推理，一张图片97s(不用cache)  810ms(cache) |
| yolov5cls           |                                                              |          |                                                              |
| mobileNet系列       |                                                              |          |                                                              |
| ResNet系列          |                                                              | ✔        | ResNet50(i3-13100CPU)+Paddle Inference推理，一张图片362ms/352ms<br/>**ResNet50(i3-13100CPU)+OenVINO推理，一张图片279ms/282ms**<br/>ResNet50(i3-13100CPU)+ONNX Runtime推理，一张图片711ms/735ms<br />ResNet50(2080tiGPU)+Paddle Inference推理，一张图片1447ms/1446ms<br/>ResNet50(2080tiGPU)+Paddle TensorRT推理，一张图片20ms/20ms<br/>ResNet50(2080tiGPU)+ONNX Runtime推理，一张图片843ms/840ms<br/>ResNet50(2080tiGPU)+Nvidia TensorRT推理，一张图片11s(不用cache)  52ms(cache) |
| **检测**            |                                                              |          |                                                              |
| yolov8              | 支持ultralytics官方的模型，但是需要转onnx。支持onnx系模型。  | ✔        | i3-13100的CPU跑yolov8s.onnx(640*640)每一帧耗时90ms-110ms<br />2080ti(GPU)跑yolov8s.onnx(640*640)每一帧耗时14ms-35ms |
| PP-yoloE            | paddle官方基于yolov3的改进版本                               |          |                                                              |
| **分割**            |                                                              |          |                                                              |
| PP-LiteSeg系列模型  | paddle官方出的超轻模型，支持pd系模型。部署时需要model.pdmodel、model.pdiparams、deploy.yaml三个文件 | ✔        | cpu测试60-90ms<br />精度一般                                 |
| U-Net系列模型       |                                                              | ✔        | cpu测试800ms<br />精度一般                                   |
| PP-HumanSeg系列模型 |                                                              |          |                                                              |
| DeepLabV3系列模型   | 151M大模型                                                   |          |                                                              |
| FCN系列模型         |                                                              |          |                                                              |
| SegFormer系列模型   |                                                              |          |                                                              |
| **OCR**             |                                                              |          |                                                              |
| pp-ocrv4            | paddle官方出的COR模型，包括文字检测、方向分类、文字识别。支持中英文。支持pd系模型(pdmodel、pdiparams)。部署时需要检测、分类、识别三个模型。 | ✔        | i3-13100CPU+Paddle Inference推理，2个字段2300ms，18个字段13000-14000ms(13-14s)<br/>i3-13100CPU+ONNX Runtime推理，2个字段50-100ms，18个字段290ms-350ms<br/>i3-13100CPU+OpenVINO推理，2个字段30ms(±5ms)，18个字段130ms(±10ms)<br/>2080tiGPU+Paddle Inference推理，2个字段60ms(±5ms) ，18个字段215ms(±5ms) <br/>2080tiGPU+ONNX Runtime推理，2个字段80ms(±5ms) ，18个字段130ms(±10ms)，不稳定，2个字段有时候会跳到200ms，18个字段会跳到800ms左右。<br/>2080tiGPU+TensorRT推理，2个字段30ms(±5ms) ，18个字段100ms(±5ms)<br/>备注：检测就一次推理，识别的话，设置的batchsize=6，所以一次推理6个字段，18个字段就要推理3次，OCR一次要识别的字段越多，耗时就越长。 |
| pp-ocrv3\v2         | 旧版本                                                       |          |                                                              |

备注✔表示已经测试跑通，❓表示fastdeploy官方宣布支持，但我们没有实际测试。同时❓表示值得做的模型。





#### third_libs

该文件夹内的文件为fastdeploy需要的第三方库文件，**使用时全部拷贝到exe所在目录**。