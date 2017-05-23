#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	int *neighbor,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;

	if (yt < ht && xt < wt && mask[curt] > 127.0f)
	{
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb*yb + xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb)
		{
			int NtIdx = wt*(yt - 1) + xt;
			int WtIdx = wt*yt + (xt - 1);
			int StIdx = wt*(yt + 1) + xt;
			int EtIdx = wt*yt + (xt + 1);

			int NbIdx = wb*(yb - 1) + xb;
			int WbIdx = wb*yb + (xb - 1);
			int SbIdx = wb*(yb + 1) + xb;
			int EbIdx = wb*yb + (xb + 1);

			int nT = 0;
			int nB = 0;
			int curt3 = curt * 3;
			fixed[curt3 + 0] = 0;
			fixed[curt3 + 1] = 0;
			fixed[curt3 + 2] = 0;
			float fT[3] = { 0, 0, 0 };
			float fB[3] = { 0, 0, 0 };
			if (yb - 1 >= 0)//N
			{
				nB++;
				if (yt - 1 >= 0)
				{
					//nB++;
					nT++;
					fT[0] += target[curt3 + 0] - target[NtIdx * 3 + 0];
					fT[1] += target[curt3 + 1] - target[NtIdx * 3 + 1];
					fT[2] += target[curt3 + 2] - target[NtIdx * 3 + 2];
					if (mask[NtIdx] < 127.0f)
					{
						fB[0] += background[NbIdx * 3 + 0];
						fB[1] += background[NbIdx * 3 + 1];
						fB[2] += background[NbIdx * 3 + 2];
					}
				}
				else
				{
					fB[0] += background[NbIdx * 3 + 0];
					fB[1] += background[NbIdx * 3 + 1];
					fB[2] += background[NbIdx * 3 + 2];
				}
			}
			if (xb - 1 >= 0)//W
			{
				nB++;
				if (xt - 1 >= 0)
				{
					//nB++;
					nT++;
					fT[0] += target[curt3 + 0] - target[WtIdx * 3 + 0];
					fT[1] += target[curt3 + 1] - target[WtIdx * 3 + 1];
					fT[2] += target[curt3 + 2] - target[WtIdx * 3 + 2];
					if (mask[WtIdx] < 127.0f)
					{
						fB[0] += background[WbIdx * 3 + 0];
						fB[1] += background[WbIdx * 3 + 1];
						fB[2] += background[WbIdx * 3 + 2];
					}
				}
				else
				{
					fB[0] += background[WbIdx * 3 + 0];
					fB[1] += background[WbIdx * 3 + 1];
					fB[2] += background[WbIdx * 3 + 2];
				}
			}
			if (yb + 1 < hb)//S
			{
				nB++;
				if (yt + 1 < ht)
				{
					//nB++;
					nT++;
					fT[0] += target[curt3 + 0] - target[StIdx * 3 + 0];
					fT[1] += target[curt3 + 1] - target[StIdx * 3 + 1];
					fT[2] += target[curt3 + 2] - target[StIdx * 3 + 2];
					if (mask[StIdx] < 127.0f)
					{
						fB[0] += background[SbIdx * 3 + 0];
						fB[1] += background[SbIdx * 3 + 1];
						fB[2] += background[SbIdx * 3 + 2];
					}
				}
				else
				{
					fB[0] += background[SbIdx * 3 + 0];
					fB[1] += background[SbIdx * 3 + 1];
					fB[2] += background[SbIdx * 3 + 2];
				}
			}
			if (xb + 1 < wb)
			{
				nB++;
				if (xt + 1 < wt)//E
				{
					//nB++;
					nT++;
					fT[0] += target[curt3 + 0] - target[EtIdx * 3 + 0];
					fT[1] += target[curt3 + 1] - target[EtIdx * 3 + 1];
					fT[2] += target[curt3 + 2] - target[EtIdx * 3 + 2];
					if (mask[EtIdx] < 127.0f)
					{
						fB[0] += background[EbIdx * 3 + 0];
						fB[1] += background[EbIdx * 3 + 1];
						fB[2] += background[EbIdx * 3 + 2];
					}
				}
				else
				{
					fB[0] += background[EbIdx * 3 + 0];
					fB[1] += background[EbIdx * 3 + 1];
					fB[2] += background[EbIdx * 3 + 2];
				}
			}
			fixed[curt3 + 0] = fT[0]/nT + fB[0]/nB;
			fixed[curt3 + 1] = fT[1]/nT + fB[1]/nB;
			fixed[curt3 + 2] = fT[2]/nT + fB[2]/nB;

			neighbor[curt] = nB;

		}
	}
}

__global__ void PoissonImageCloningIteration(
	float *fixed,
	const float *mask,
	float *buffer1,
	float *buffer2,
	int *neighbor,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;

	if (yt < ht && xt < wt && mask[curt] > 127.0f)
	{
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb*yb + xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb)
		{
			int NtIdx = wt*(yt - 1) + xt;
			int WtIdx = wt*yt + (xt - 1);
			int StIdx = wt*(yt + 1) + xt;
			int EtIdx = wt*yt + (xt + 1);

			float next[3] = { 0, 0, 0 };
			if (yb - 1 >= 0)//N
			{
				if (yt - 1 >= 0 && mask[NtIdx] > 127.0f)
				{
					next[0] += buffer1[NtIdx * 3 + 0];
					next[1] += buffer1[NtIdx * 3 + 1];
					next[2] += buffer1[NtIdx * 3 + 2];
				}
			}

			if (xb - 1 >= 0)//W
			{
				if (xt - 1 >= 0 && mask[WtIdx] > 127.0f)
				{
					next[0] += buffer1[WtIdx * 3 + 0];
					next[1] += buffer1[WtIdx * 3 + 1];
					next[2] += buffer1[WtIdx * 3 + 2];
				}
			}
			if (yb + 1 < hb)//S
			{
				if (yt + 1 < ht && mask[StIdx] > 127.0f)
				{
					next[0] += buffer1[StIdx * 3 + 0];
					next[1] += buffer1[StIdx * 3 + 1];
					next[2] += buffer1[StIdx * 3 + 2];
				}
			}
			if (xb + 1 < wb)//E
			{
				if (xt + 1 < wt && mask[EtIdx] > 127.0f)
				{
					next[0] += buffer1[EtIdx * 3 + 0];
					next[1] += buffer1[EtIdx * 3 + 1];
					next[2] += buffer1[EtIdx * 3 + 2];
				}
			}
			buffer2[curt * 3 + 0] = next[0] / (float)neighbor[curt] + fixed[curt * 3 + 0];
			buffer2[curt * 3 + 1] = next[1] / (float)neighbor[curt] + fixed[curt * 3 + 1];
			buffer2[curt * 3 + 2] = next[2] / (float)neighbor[curt] + fixed[curt * 3 + 2];
		}
	}
}

__global__ void Clone(
	const float *background,
	float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;

	if (yt < ht && xt < wt && mask[curt] > 127.0f)
	{
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb*yb + xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb)
		{
			output[curb * 3 + 0] = target[curt * 3 + 0];
			output[curb * 3 + 1] = target[curt * 3 + 1];
			output[curb * 3 + 2] = target[curt * 3 + 2];
		}
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
	
	float *fixed, *buffer1, *buffer2;
	cudaMalloc((void**)&fixed, 3 * wt * ht * sizeof(float));
	cudaMalloc((void**)&buffer1, 3 * wt * ht * sizeof(float));
	cudaMalloc((void**)&buffer2, 3 * wt * ht * sizeof(float));
	int *neighbor;
	cudaMalloc((void**)&neighbor, wt * ht * sizeof(int));

	CalculateFixed << <gdim, bdim >> >(
		background, target, mask, fixed, neighbor,
		wb, hb, wt, ht, oy, ox);
	cudaMemcpy(buffer1, target, sizeof(float) * 3 * wt * ht, cudaMemcpyDeviceToDevice);
	for (int i = 0; i < 10000; ++i) {
		PoissonImageCloningIteration << <gdim, bdim >> >(
			fixed, mask, buffer1, buffer2, neighbor, wb, hb, wt, ht, oy, ox);
		cudaThreadSynchronize();
		PoissonImageCloningIteration << <gdim, bdim >> >(
			fixed, mask, buffer2, buffer1, neighbor, wb, hb, wt, ht, oy, ox);
		cudaThreadSynchronize();
	}

	cudaMemcpy(output, background, wb*hb*sizeof(float) * 3, cudaMemcpyDeviceToDevice);

	Clone << <gdim, bdim>> >(
		background, buffer1, mask, output,
		wb, hb, wt, ht, oy, ox);

	cudaFree(fixed);
	cudaFree(buffer1);
	cudaFree(buffer2);
	cudaFree(neighbor);
}
