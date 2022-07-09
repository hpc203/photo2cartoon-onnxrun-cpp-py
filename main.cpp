#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>


using namespace cv;
using namespace std;
using namespace Ort;

class photo2cartoon
{
public:
	photo2cartoon();
	Mat inference(Mat cv_image);
private:

	Mat preprocess_head_seg(Mat srcimg);
	const int inpWidth = 384;   
	const int inpHeight = 384;   ///高宽也可以从onnx文件里获得
	vector<float> input_image_;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "head_seg");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> output_node_dims; // >=1 outputs

	vector<float> input_cartoon;
	Env env_cartoon = Env(ORT_LOGGING_LEVEL_ERROR, "cartoon");
	Ort::Session *ort_session_cartoon = nullptr;
	SessionOptions sessionOptions_cartoon = SessionOptions();
	vector<char*> input_names_cartoon;
	vector<char*> output_names_cartoon;
	vector<vector<int64_t>> output_node_dims_cartoon;
	const int inpWidth_cartoon = 256;
	const int inpHeight_cartoon = 256;    ///高宽也可以从onnx文件里获得
	void preprocess_cartoon(Mat img);
	Mat generate_cartoon(Mat image, Mat mask_rs);
};

photo2cartoon::photo2cartoon()
{
	string model_path = "photo2cartoon_models/minivision_head_seg.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
	///OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释

	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
	////ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}

	model_path = "photo2cartoon_models/minivision_female_photo2cartoon.onnx";
	std::wstring widestr_cartoon = std::wstring(model_path.begin(), model_path.end());  ////windows写法
	///OrtStatus* status_cartoon = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions_cartoon, 0);  ///如果使用cuda加速，需要取消注释

	sessionOptions_cartoon.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session_cartoon = new Session(env_cartoon, widestr_cartoon.c_str(), sessionOptions_cartoon);  ////windows写法
	////ort_session_cartoon = new Session(env_cartoon, model_path.c_str(), sessionOptions_cartoon);  ////linux写法

	numInputNodes = ort_session_cartoon->GetInputCount();
	numOutputNodes = ort_session_cartoon->GetOutputCount();
	AllocatorWithDefaultOptions allocator_cartoon;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names_cartoon.push_back(ort_session_cartoon->GetInputName(i, allocator_cartoon));
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names_cartoon.push_back(ort_session_cartoon->GetOutputName(i, allocator_cartoon));
		Ort::TypeInfo output_type_info_cartoon = ort_session_cartoon->GetOutputTypeInfo(i);
		auto output_tensor_info_cartoon = output_type_info_cartoon.GetTensorTypeAndShapeInfo();
		auto output_dims_cartoon = output_tensor_info_cartoon.GetShape();
		output_node_dims_cartoon.push_back(output_dims_cartoon);
	}
}

Mat photo2cartoon::preprocess_head_seg(Mat srcimg)
{
	Mat rgbimg;
	cvtColor(srcimg, rgbimg, COLOR_BGR2RGB);
	Mat dstimg;
	resize(rgbimg, dstimg, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);
	
	int row = dstimg.rows;
	int col = dstimg.cols;
	this->input_image_.resize(row * col * dstimg.channels());
	int k = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			for (int c = 0; c < 3; c++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[k] = pix / 255.0;
				k++;
			}
		}
	}
	return rgbimg;
}

void photo2cartoon::preprocess_cartoon(Mat img)
{
	int row = this->inpHeight_cartoon;
	int col = this->inpWidth_cartoon;
	this->input_cartoon.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<float>(i)[j * 3 + c];
				this->input_cartoon[c * row * col + i * col + j] = pix;
			}
		}
	}
}

Mat photo2cartoon::generate_cartoon(Mat mat_merged_rs, Mat mask_rs)
{
	this->preprocess_cartoon(mat_merged_rs);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight_cartoon, this->inpWidth_cartoon };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_cartoon.data(), input_cartoon.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session_cartoon->Run(RunOptions{ nullptr }, &input_names_cartoon[0], &input_tensor_, 1, output_names_cartoon.data(), output_names_cartoon.size());

	Ort::Value &cartoon_pred = ort_outputs.at(0); // (1,3,256,256)
	auto cartoon_dims = cartoon_pred.GetTensorTypeAndShapeInfo().GetShape();
	const unsigned int out_h = cartoon_dims.at(2);
	const unsigned int out_w = cartoon_dims.at(3);
	const unsigned int channel_step = out_h * out_w;
	const unsigned int mask_h = mask_rs.rows;
	const unsigned int mask_w = mask_rs.cols;
	// fast assign & channel transpose(CHW->HWC).
	float *cartoon_ptr = cartoon_pred.GetTensorMutableData<float>();

	vector<Mat> cartoon_channel_mats;
	Mat rmat(out_h, out_w, CV_32FC1, cartoon_ptr); // R
	Mat gmat(out_h, out_w, CV_32FC1, cartoon_ptr + channel_step); // G
	Mat bmat(out_h, out_w, CV_32FC1, cartoon_ptr + 2 * channel_step); // B
	rmat = (rmat + 1.f) * 127.5f;
	gmat = (gmat + 1.f) * 127.5f;
	bmat = (bmat + 1.f) * 127.5f;
	cartoon_channel_mats.push_back(rmat);
	cartoon_channel_mats.push_back(gmat);
	cartoon_channel_mats.push_back(bmat);
	Mat cartoon;
	merge(cartoon_channel_mats, cartoon); // CV_32FC3 allocated
	if (out_h != mask_h || out_w != mask_w)
		resize(cartoon, cartoon, Size(mask_w, mask_h));
	// combine & RGB -> BGR -> uint8
	cartoon = cartoon.mul(mask_rs) + (1.f - mask_rs) * 255.f;
	cvtColor(cartoon, cartoon, COLOR_RGB2BGR);
	cartoon.convertTo(cartoon, CV_8UC3);
	return cartoon;
}

Mat photo2cartoon::inference(Mat srcimg)
{
	Mat rgbimg = this->preprocess_head_seg(srcimg);
	array<int64_t, 4> input_shape_{ 1, this->inpHeight, this->inpWidth, 3};

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names.data(), &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	// 3. post process.
	Value &mask_pred = ort_outputs.at(0); // (1,384,384,1)
	auto mask_dims = mask_pred.GetTensorTypeAndShapeInfo().GetShape();
	const unsigned int out_h = mask_dims.at(1);
	const unsigned int out_w = mask_dims.at(2);
	float *mask_ptr = mask_pred.GetTensorMutableData<float>();

	Mat mask;
	Mat mask_out(out_h, out_w, CV_32FC1, mask_ptr); 
	resize(mask_out, mask, Size(srcimg.cols, srcimg.rows)); 
	const unsigned int mask_channels = mask.channels();

	Mat face;
	resize(rgbimg, face, Size(this->inpWidth_cartoon, this->inpHeight_cartoon));
	Mat mask_rs;;
	resize(mask, mask_rs, Size(this->inpWidth_cartoon, this->inpHeight_cartoon));
	if (mask_channels != 3) cvtColor(mask_rs, mask_rs, COLOR_GRAY2BGR); // CV_32FC3
	face.convertTo(face, CV_32FC3, 1.f, 0.f); // CV_32FC3
	Mat mat_merged_rs = face.mul(mask_rs) + (1.f - mask_rs) * 255.f;
	mat_merged_rs.convertTo(mat_merged_rs, CV_32FC3, 1.f / 127.5f, -1.f);
	
	Mat dstimg = this->generate_cartoon(mat_merged_rs, mask_rs);
	return dstimg;
}

int main()
{
	photo2cartoon mynet;
	string imgpath = "testimgs/1.jpg";
	Mat srcimg = imread(imgpath);
	Mat dstimg = mynet.inference(srcimg);

	namedWindow("srcimg", WINDOW_NORMAL);
	imshow("srcimg", srcimg);
	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, dstimg);
	waitKey(0);
	destroyAllWindows();
}