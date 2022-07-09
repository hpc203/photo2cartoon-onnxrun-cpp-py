# photo2cartoon-onnxrun-cpp-py
使用ONNXRuntime部署StyleGAN人像卡通画，包含C++和Python两个版本的程序
。起初我是想用opencv做部署的，可是opencv的dnn模块读取onnx文件出错了，无赖只能使用onnxruntime部署。
本套程序的输入图片是人像大头照，如果输入图片里包含太多背景，需要先做人脸检测+人脸矫正。
裁剪出人像区域后，作为本套程序的输入图像，否则效果会大打折扣。

onnx文件，在百度云盘，下载链接：https://pan.baidu.com/s/1FiPHQHZ6VbIyTTBe8qeJXg 
提取码：fdzj
