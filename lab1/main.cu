#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "lab1.h"

#ifdef Debug
#include "cv.h"
#include "highgui.h"
#endif
using namespace std;

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

void YUV4202RGB(int W, int H, const uint8_t *yuv, uint8_t *rgb)
{
	int size = W * H;
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			float Y = yuv[i * W + j];
			float U = yuv[(int)(size + (i / 2)*(W / 2) + j / 2)];
			float V = yuv[(int)(size * 1.25 + (i / 2)*(W / 2) + j / 2)];

			float R = Y + 1.402 * (V - 128);
			float G = Y - 0.344 * (U - 128) - 0.714 * (V - 128);
			float B = Y + 1.772 * (U - 128);
			
			rgb[(i * W + j) * 3] = R;
			rgb[(i * W + j) * 3 + 1] = G;
			rgb[(i * W + j) * 3 + 2] = B;
		}
	}
}

void RGB2YUV420(int W, int H, const uint8_t *rgb, uint8_t *yuv)
{
	int size = W * H;
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			float R = rgb[(i * W + j) * 3];
			float G = rgb[(i * W + j) * 3 + 1];
			float B = rgb[(i * W + j) * 3 + 2];

			float Y = 0.299 * R + 0.587 * G + 0.114 * B;
			float U = -0.169 * R + -0.331 * G + 0.5 * B + 128;
			float V = 0.5 * R + -0.419 * G + -0.081 * B + 128;
			
			yuv[i * W + j] = Y;
			yuv[(int)(size + (i / 2)*(W / 2) + j / 2)] = U;
			yuv[(int)(size * 1.25 + (i / 2)*(W / 2) + j / 2)] = V;
		}
	}
}


int main(int argc, char **argv)
{
	Lab1VideoGenerator g;
	Lab1VideoInfo i;

	g.get_info(i);
	if (i.w == 0 || i.h == 0 || i.n_frame == 0 || i.fps_n == 0 || i.fps_d == 0) {
		puts("Cannot be zero");
		abort();
	} else if (i.w%2 != 0 || i.h%2 != 0) {
		puts("Only even frame size is supported");
		abort();
	}
	unsigned FRAME_SIZE = i.w*i.h;
	//unsigned FRAME_SIZE = i.w*i.h * 3/2;
	MemoryBuffer<uint8_t> frameb(FRAME_SIZE * 3/2);
	auto frames = frameb.CreateSync(FRAME_SIZE * 3/2);
	FILE *fp = fopen("result.y4m", "wb");
	fprintf(fp, "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 C420\n", i.w, i.h, i.fps_n, i.fps_d);

	//uint8_t *YUV420Data = (uint8_t*)malloc(sizeof(uint8_t) * i.w * i.h * 3 / 2);
	//uint8_t *rgbData = (uint8_t*)malloc(sizeof(uint8_t) * i.w * i.h * 3);
	for (unsigned j = 0; j < i.n_frame; ++j) {
		fputs("FRAME\n", fp);
		g.Generate(frames.get_gpu_wo());
#ifdef Debug
		rgbData = (uint8_t*)malloc(sizeof(uint8_t) * i.w * i.h * 3);
		YUV4202RGB(i.w, i.h, frames.get_cpu_ro(), rgbData);
		//uint8_t *YUV420Data = (uint8_t*)malloc(sizeof(uint8_t) * i.w * i.h * 3 / 2);
		//RGB2YUV420(i.w, i.h, frames.get_cpu_ro(), YUV420Data);
		//cv::Mat img(i.h, i.w, CV_8UC3, rgbData);
		//RGB2YUV420(i.w, i.h, frames.get_cpu_ro(), YUV420Data);
		cv::Mat img(i.h, i.w, CV_8UC3, rgbData);
		cv::cvtColor(img, img, CV_RGB2BGR);
		cv::imshow("TEST", img);
		cv::waitKey(1.0/24.0 * 100);
#else
		//RGB2YUV420(i.w, i.h, frames.get_cpu_ro(), YUV420Data);
		fwrite(frames.get_cpu_ro(), sizeof(uint8_t), FRAME_SIZE * 3 / 2, fp);
#endif
	}
	//free(rgbData);
	//free(YUV420Data);
	fclose(fp);
	system("pause");
	return 0;
}

