#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 1024

__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct notSpace
{//\n 10
	__host__ __device__ const char &operator()(const __int8 &x) const { return (x != 10) ? 1 : 0; }
};
void CountPosition1(const char *text, int *pos, int text_size)
{
	thrust::device_ptr<char> charArray((char *)text);
	thrust::device_ptr<int> intArray(pos);
	thrust::transform(thrust::device, charArray, charArray + text_size, intArray, notSpace());
	thrust::inclusive_scan_by_key(thrust::device, intArray, intArray + text_size, intArray, intArray);
}

__device__ int MaxPow2(int i)
{
	int p2 = 1;
	while (i >= p2 * 2){ p2 *= 2; }
	return p2;
}

__device__ int MinPow2M1(int i)
{
	int p2 = 1;
	while (i >= p2){ p2 *= 2; }
	return p2 - 1;
}

__device__ int PowD(int base, int times)
{
	int r = 1;
	while (times-- > 0) r *= base;
	return r;
}

__global__ void ScanButtonUp(const char *text, int *pos, int text_size, int *flagsDevice, int level)
{

	int i = (blockIdx.x * blockDim.x + threadIdx.x + 1) * PowD(BLOCK_SIZE, level) - 1;
	int idx = threadIdx.x;

	if (i < text_size)
	{
		__shared__ int sText[BLOCK_SIZE];
		__shared__ int flag[BLOCK_SIZE];//-1 right 0, 0 right 1, 1 both 1

		if (level == 0)
		{
			if (text[i] != '\n')
			{
				sText[idx] = 1;
				flag[idx] = 1;
			}
			else
			{
				sText[idx] = 0;
				flag[idx] = -1;
			}

		}
		else
		{
			sText[idx] = pos[i];
			flag[idx] = flagsDevice[i];
		}
		syncthreads();
		int idxS;

		if ((idx + 1) % 2 == 0)
		{
			idxS = idx - 1;
			if (flag[idx] == 1 && flag[idxS] >= 0)
				sText[idx] += sText[idxS];
			flag[idx] = (flag[idx] == 1) ? (flag[idxS] >= 0) ? flag[idxS] : 0 : flag[idx];
		}
		syncthreads();

		if ((idx + 1) % 4 == 0)
		{
			idxS = idx - 2;
			if (flag[idx] == 1 && flag[idxS] >= 0)
				sText[idx] += sText[idxS];
			flag[idx] = (flag[idx] == 1) ? (flag[idxS] >= 0) ? flag[idxS] : 0 : flag[idx];
		}
		syncthreads();

		if ((idx + 1) % 8 == 0)
		{
			idxS = idx - 4;
			if (flag[idx] == 1 && flag[idxS] >= 0)
				sText[idx] += sText[idxS];
			flag[idx] = (flag[idx] == 1) ? (flag[idxS] >= 0) ? flag[idxS] : 0 : flag[idx];
		}
		syncthreads();

		if ((idx + 1) % 16 == 0)
		{
			idxS = idx - 8;
			if (flag[idx] == 1 && flag[idxS] >= 0)
				sText[idx] += sText[idxS];
			flag[idx] = (flag[idx] == 1) ? (flag[idxS] >= 0) ? flag[idxS] : 0 : flag[idx];
		}
		syncthreads();

		if ((idx + 1) % 32 == 0)
		{
			idxS = idx - 16;
			if (flag[idx] == 1 && flag[idxS] >= 0)
				sText[idx] += sText[idxS];
			flag[idx] = (flag[idx] == 1) ? (flag[idxS] >= 0) ? flag[idxS] : 0 : flag[idx];
		}
		syncthreads();

		if ((idx + 1) % 64 == 0)
		{
			idxS = idx - 32;
			if (flag[idx] == 1 && flag[idxS] >= 0)
				sText[idx] += sText[idxS];
			flag[idx] = (flag[idx] == 1) ? (flag[idxS] >= 0) ? flag[idxS] : 0 : flag[idx];
		}
		syncthreads();
		if ((idx + 1) % 128 == 0)
		{
			idxS = idx - 64;
			if (flag[idx] == 1 && flag[idxS] >= 0)
				sText[idx] += sText[idxS];
			flag[idx] = (flag[idx] == 1) ? (flag[idxS] >= 0) ? flag[idxS] : 0 : flag[idx];
		}
		syncthreads();
		if ((idx + 1) % 256 == 0)
		{
			idxS = idx - 128;
			if (flag[idx] == 1 && flag[idxS] >= 0)
				sText[idx] += sText[idxS];
			flag[idx] = (flag[idx] == 1) ? (flag[idxS] >= 0) ? flag[idxS] : 0 : flag[idx];
		}
		syncthreads();
		if ((idx + 1) % 512 == 0)
		{
			idxS = idx - 256;
			if (flag[idx] == 1 && flag[idxS] >= 0)
				sText[idx] += sText[idxS];
			flag[idx] = (flag[idx] == 1) ? (flag[idxS] >= 0) ? flag[idxS] : 0 : flag[idx];
		}
		syncthreads();
		if ((idx + 1) % 1024 == 0)
		{
			idxS = idx - 512;
			if (flag[idx] == 1 && flag[idxS] >= 0)
				sText[idx] += sText[idxS];
			flag[idx] = (flag[idx] == 1) ? (flag[idxS] >= 0) ? flag[idxS] : 0 : flag[idx];
		}
		syncthreads();

		pos[i] = sText[idx];
		flagsDevice[i] = flag[idx];


	}

}

__global__ void ScanTopDown(const char *text, int *pos, int text_size, int *flagsDevice, int level, int toplevel, bool *endend, bool *nextSwitch, int *switchIdx, int *posSub)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x + 1) * PowD(BLOCK_SIZE, level) - 1;
	if (i + 1 > text_size && text_size > i - PowD(BLOCK_SIZE, level) + 1)
		i = text_size - 1;
	int idx = threadIdx.x;

	if (i < text_size)
	{
		__shared__ int sText[BLOCK_SIZE];
		__shared__ int flag[BLOCK_SIZE];//-1 right 0, 0 right 1, 1 both 1

		sText[idx] = pos[i];
		flag[idx] = flagsDevice[i];
		syncthreads();

		int idxS;

		int mP2M1_idx = MinPow2M1(idx);
		int MP2_idx = MaxPow2(idx);
		int mP2M1_i = MinPow2M1(i);
		int MP2_i = MaxPow2(i);

		if ((i + 1) == text_size)
		{
			if (level == toplevel - 1)
			{
				*endend = false;
				*nextSwitch = false;
				sText[idx] = sText[MP2_idx - 1];
				flag[idx] = flag[MP2_idx - 1];

				sText[MP2_idx - 1] = 0;
				flag[MP2_idx - 1] = 1;

				if (i - (MP2_i - 1) <= 1)
					*endend = true;

				if (idx > (MP2_idx - 1) + MP2_idx / 2 && idx > 1)
				{
					*switchIdx = (MP2_idx - 1) + MP2_idx / 2;
					*nextSwitch = true;
				}
				else
				{
					*switchIdx = MP2_idx - 1;
					*nextSwitch = false;
				}
			}
			else
			{
				if (idx > MP2_idx / 2)
				{
					*switchIdx = MP2_idx - 1;
					*nextSwitch = true;
				}
				else
				{
					*switchIdx = -1;
					*nextSwitch = false;
				}
			}
		}
		syncthreads();
		
		if ((idx + 1) % 1024 == 0 && (i + 1) < text_size)
		{
			idxS = idx - 512;
			int tS = sText[idx];
			int tF = flag[idx];

			if (flag[idxS] == 1 && flag[idx] >= 0)
				sText[idx] += sText[idxS];
			else
				sText[idx] = sText[idxS];
			flag[idx] = (flag[idxS] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[idxS];

			sText[idxS] = tS;
			flag[idxS] = tF;
		}
		else if (!*endend && (i + 1) == text_size)
		{
			if (*nextSwitch)
			{
				int tS = sText[idx];
				int tF = flag[idx];
				if (flag[*switchIdx] == 1 && flag[idx] >= 0)
					sText[idx] += sText[*switchIdx];
				else
					sText[idx] = sText[*switchIdx];
				flag[idx] = (flag[*switchIdx] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[*switchIdx];

				sText[*switchIdx] = tS;
				flag[*switchIdx] = tF;
				if (i - (*switchIdx * PowD(BLOCK_SIZE, level)) <= 1)
					*endend = true;
			}

			if (idx > *switchIdx + 256)
			{
				*switchIdx = *switchIdx + 256;
				*nextSwitch = true;
			}
			else
				*nextSwitch = false;
		}
		syncthreads();
		if ((idx + 1) % 512 == 0 && (i + 1) < text_size)
		{
			idxS = idx - 256;
			int tS = sText[idx];
			int tF = flag[idx];

			if (flag[idxS] == 1 && flag[idx] >= 0)
				sText[idx] += sText[idxS];
			else
				sText[idx] = sText[idxS];
			flag[idx] = (flag[idxS] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[idxS];

			sText[idxS] = tS;
			flag[idxS] = tF;
		}
		else if (!*endend && (i + 1) == text_size)
		{
			if (*nextSwitch)
			{
				int tS = sText[idx];
				int tF = flag[idx];
				if (flag[*switchIdx] == 1 && flag[idx] >= 0)
					sText[idx] += sText[*switchIdx];
				else
					sText[idx] = sText[*switchIdx];
				flag[idx] = (flag[*switchIdx] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[*switchIdx];

				sText[*switchIdx] = tS;
				flag[*switchIdx] = tF;
				if (i - (*switchIdx * PowD(BLOCK_SIZE, level)) <= 1)
					*endend = true;
			}

			if (idx > *switchIdx + 128)
			{
				*switchIdx = *switchIdx + 128;
				*nextSwitch = true;
			}
			else
				*nextSwitch = false;
		}
		syncthreads();
		if ((idx + 1) % 256 == 0 && (i + 1) < text_size)
		{
			idxS = idx - 128;
			int tS = sText[idx];
			int tF = flag[idx];

			if (flag[idxS] == 1 && flag[idx] >= 0)
				sText[idx] += sText[idxS];
			else
				sText[idx] = sText[idxS];
			flag[idx] = (flag[idxS] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[idxS];

			sText[idxS] = tS;
			flag[idxS] = tF;
		}
		else if (!*endend && (i + 1) == text_size)
		{
			if (*nextSwitch)
			{
				int tS = sText[idx];
				int tF = flag[idx];

				if (flag[*switchIdx] == 1 && flag[idx] >= 0)
					sText[idx] += sText[*switchIdx];
				else
					sText[idx] = sText[*switchIdx];
				flag[idx] = (flag[*switchIdx] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[*switchIdx];

				sText[*switchIdx] = tS;
				flag[*switchIdx] = tF;
				if (i - (*switchIdx * PowD(BLOCK_SIZE, level)) <= 1)
					*endend = true;
			}

			if (idx > *switchIdx + 64)
			{
				*switchIdx = *switchIdx + 64;
				*nextSwitch = true;
			}
			else
				*nextSwitch = false;
		}
		syncthreads();
		if ((idx + 1) % 128 == 0 && (i + 1) < text_size)
		{
			idxS = idx - 64;
			int tS = sText[idx];
			int tF = flag[idx];

			if (flag[idxS] == 1 && flag[idx] >= 0)
				sText[idx] += sText[idxS];
			else
				sText[idx] = sText[idxS];
			flag[idx] = (flag[idxS] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[idxS];

			sText[idxS] = tS;
			flag[idxS] = tF;
		}
		else if (!*endend && (i + 1) == text_size)
		{
			if (*nextSwitch)
			{
				int tS = sText[idx];
				int tF = flag[idx];

				if (flag[*switchIdx] == 1 && flag[idx] >= 0)
					sText[idx] += sText[*switchIdx];
				else
					sText[idx] = sText[*switchIdx];
				flag[idx] = (flag[*switchIdx] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[*switchIdx];

				sText[*switchIdx] = tS;
				flag[*switchIdx] = tF;
				if (i - (*switchIdx * PowD(BLOCK_SIZE, level)) <= 1)
					*endend = true;
			}

			if (idx > *switchIdx + 32)
			{
				*switchIdx = *switchIdx + 32;
				*nextSwitch = true;
			}
			else
				*nextSwitch = false;
		}
		syncthreads();
		if ((idx + 1) % 64 == 0 && (i + 1) < text_size)
		{
			idxS = idx - 32;
			int tS = sText[idx];
			int tF = flag[idx];

			if (flag[idxS] == 1 && flag[idx] >= 0)
				sText[idx] += sText[idxS];
			else
				sText[idx] = sText[idxS];
			flag[idx] = (flag[idxS] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[idxS];

			sText[idxS] = tS;
			flag[idxS] = tF;
		}
		else if (!*endend && (i + 1) == text_size)
		{
			if (*nextSwitch)
			{
				int tS = sText[idx];
				int tF = flag[idx];

				if (flag[*switchIdx] == 1 && flag[idx] >= 0)
					sText[idx] += sText[*switchIdx];
				else
					sText[idx] = sText[*switchIdx];
				flag[idx] = (flag[*switchIdx] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[*switchIdx];

				sText[*switchIdx] = tS;
				flag[*switchIdx] = tF;
				if (i - (*switchIdx * PowD(BLOCK_SIZE, level)) <= 1)
					*endend = true;
			}

			if (idx > *switchIdx + 16)
			{
				*switchIdx = *switchIdx + 16;
				*nextSwitch = true;
			}
			else
				*nextSwitch = false;
		}
		syncthreads();
		if ((idx + 1) % 32 == 0 && (i + 1) < text_size)
		{
			idxS = idx - 16;
			int tS = sText[idx];
			int tF = flag[idx];

			if (flag[idxS] == 1 && flag[idx] >= 0)
				sText[idx] += sText[idxS];
			else
				sText[idx] = sText[idxS];
			flag[idx] = (flag[idxS] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[idxS];

			sText[idxS] = tS;
			flag[idxS] = tF;
		}
		else if (!*endend && (i + 1) == text_size)
		{
			if (*nextSwitch)
			{
				int tS = sText[idx];
				int tF = flag[idx];

				if (flag[*switchIdx] == 1 && flag[idx] >= 0)
					sText[idx] += sText[*switchIdx];
				else
					sText[idx] = sText[*switchIdx];
				flag[idx] = (flag[*switchIdx] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[*switchIdx];

				sText[*switchIdx] = tS;
				flag[*switchIdx] = tF;
				if (i - (*switchIdx * PowD(BLOCK_SIZE, level)) <= 1)
					*endend = true;
			}

			if (idx > *switchIdx + 8)
			{
				*switchIdx = *switchIdx + 8;
				*nextSwitch = true;
			}
			else
				*nextSwitch = false;
		}
		syncthreads();
		if ((idx + 1) % 16 == 0 && (i + 1) < text_size)
		{
			idxS = idx - 8;
			int tS = sText[idx];
			int tF = flag[idx];

			if (flag[idxS] == 1 && flag[idx] >= 0)
				sText[idx] += sText[idxS];
			else
				sText[idx] = sText[idxS];
			flag[idx] = (flag[idxS] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[idxS];

			sText[idxS] = tS;
			flag[idxS] = tF;
		}
		else if (!*endend && (i + 1) == text_size)
		{
			if (*nextSwitch)
			{
				int tS = sText[idx];
				int tF = flag[idx];

				if (flag[*switchIdx] == 1 && flag[idx] >= 0)
					sText[idx] += sText[*switchIdx];
				else
					sText[idx] = sText[*switchIdx];
				flag[idx] = (flag[*switchIdx] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[*switchIdx];

				sText[*switchIdx] = tS;
				flag[*switchIdx] = tF;
				if (i - (*switchIdx * PowD(BLOCK_SIZE, level)) <= 1)
					*endend = true;
			}

			if (idx > *switchIdx + 4)
			{
				*switchIdx = *switchIdx + 4;
				*nextSwitch = true;
			}
			else
				*nextSwitch = false;
		}
		syncthreads();
		if ((idx + 1) % 8 == 0 && (i + 1) < text_size)
		{
			idxS = idx - 4;
			int tS = sText[idx];
			int tF = flag[idx];

			if (flag[idxS] == 1 && flag[idx] >= 0)
				sText[idx] += sText[idxS];
			else
				sText[idx] = sText[idxS];
			flag[idx] = (flag[idxS] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[idxS];

			sText[idxS] = tS;
			flag[idxS] = tF;
		}
		else if (!*endend && (i + 1) == text_size)
		{
			if (*nextSwitch)
			{
				int tS = sText[idx];
				int tF = flag[idx];

				if (flag[*switchIdx] == 1 && flag[idx] >= 0)
					sText[idx] += sText[*switchIdx];
				else
					sText[idx] = sText[*switchIdx];
				flag[idx] = (flag[*switchIdx] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[*switchIdx];

				sText[*switchIdx] = tS;
				flag[*switchIdx] = tF;
				if (i - (*switchIdx * PowD(BLOCK_SIZE, level)) <= 1)
					*endend = true;

			}

			if (idx > *switchIdx + 2)
			{
				*switchIdx = *switchIdx + 2;
				*nextSwitch = true;
			}
			else
				*nextSwitch = false;
		}
		syncthreads();
		if ((idx + 1) % 4 == 0 && (i + 1) < text_size)
		{
			idxS = idx - 2;

			int tS = sText[idx];
			int tF = flag[idx];

			if (flag[idxS] == 1 && flag[idx] >= 0)
				sText[idx] += sText[idxS];
			else
				sText[idx] = sText[idxS];
			flag[idx] = (flag[idxS] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[idxS];

			sText[idxS] = tS;
			flag[idxS] = tF;
		}
		else if (!*endend && (i + 1) == text_size)
		{
			if (*nextSwitch)
			{
				int tS = sText[idx];
				int tF = flag[idx];

				if (flag[*switchIdx] == 1 && flag[idx] >= 0)
					sText[idx] += sText[*switchIdx];
				else
					sText[idx] = sText[*switchIdx];
				flag[idx] = (flag[*switchIdx] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[*switchIdx];

				sText[*switchIdx] = tS;
				flag[*switchIdx] = tF;
				if (i - (*switchIdx * PowD(BLOCK_SIZE, level)) <= 1)
					*endend = true;

			}

			if (idx > *switchIdx + 1)
			{
				*switchIdx = *switchIdx + 1;
				*nextSwitch = true;
			}
			else
				*nextSwitch = false;
		}
		syncthreads();
		if ((idx + 1) % 2 == 0 && (i + 1) < text_size)
		{
			idxS = idx - 1;

			int tS = sText[idx];
			int tF = flag[idx];

			if (flag[idxS] == 1 && flag[idx] >= 0)
				sText[idx] += sText[idxS];
			else
				sText[idx] = sText[idxS];
			flag[idx] = (flag[idxS] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[idxS];

			sText[idxS] = tS;
			flag[idxS] = tF;
		}
		else if (!*endend && (i + 1) == text_size)
		{
			if (*nextSwitch)
			{
				int tS = sText[idx];
				int tF = flag[idx];

				if (flag[*switchIdx] == 1 && flag[idx] >= 0)
					sText[idx] += sText[*switchIdx];
				else
					sText[idx] = sText[*switchIdx];
				flag[idx] = (flag[*switchIdx] == 1) ? (flag[idx] >= 0) ? flag[idx] : 0 : flag[*switchIdx];

				sText[*switchIdx] = tS;
				flag[*switchIdx] = tF;
				if (i - (*switchIdx * PowD(BLOCK_SIZE, level)) <= 1)
					*endend = true;
			}
		}
		syncthreads();
		pos[i] = sText[idx];
		flagsDevice[i] = flag[idx];

		if (level == 0)
		{
			if (idx == 0 && i != 0)
				posSub[blockIdx.x - 1] = sText[idx];

			if (i != text_size - 1)
			{
				if (idx != BLOCK_SIZE - 1)
				{
					pos[i] = sText[idx + 1];
				}
			}
			else
			{
				pos[i] = (text[i] != '\n') ? sText[idx] + 1 : 0;
			}
		}
	}

}

__global__ void ScanShift(int *pos, int *posSub, int subSize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < subSize)
	{
		pos[(i + 1) * BLOCK_SIZE - 1] = posSub[i];
	}
}

__global__ void printD(int *pos, int text_size, int level)
{
	if (threadIdx.x == 0)
	{
		printf("%d %d\n", level, text_size);
		for (int i = 0; i < text_size; i++)
		{
			printf("%2d ", pos[i]);
			if ((i + 1) % 32 == 0)
				printf(" \\ ");
			else if ((i + 1) % 16 == 0)
				printf(" ! ");
			else if ((i + 1) % 8 == 0)
				printf(" | ");
			else if ((i + 1) % 4 == 0)
				printf(" ? ");
			else if ((i + 1) % 2 == 0)
				printf(" : ");
		}
		printf("\n\n");
	}
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	int *flagsDevice, *posSub;
	cudaMalloc((void**)&flagsDevice, sizeof(int) * text_size);

	int totalLevel = ceil(log(text_size) / log(BLOCK_SIZE));
	int blockNum = ((text_size - 1) / BLOCK_SIZE + 1);
	int subSize = blockNum - 1;
	cudaMalloc((void**)&posSub, sizeof(int) * subSize);

	int *blockNumArray = (int*)malloc(sizeof(int) * totalLevel);
	for (int level = 0; level < totalLevel; level++)
	{
		blockNumArray[level] = blockNum;
		ScanButtonUp << <blockNum, BLOCK_SIZE >> >(text, pos, text_size, flagsDevice, level);
		blockNum = ((blockNum - 1) / BLOCK_SIZE + 1);
	}
	int *switchIdx;
	cudaMalloc((void**)&switchIdx, sizeof(int));
	bool *endend, *nextSwitch;
	cudaMalloc((void**)&endend, sizeof(bool));
	cudaMalloc((void**)&nextSwitch, sizeof(bool));
	for (int level = totalLevel - 1; level >= 0; level--)
	{
		blockNum = blockNumArray[level];
		ScanTopDown << <blockNum, BLOCK_SIZE >> >(text, pos, text_size, flagsDevice, level, totalLevel, endend, nextSwitch, switchIdx, posSub);
	}
	ScanShift << < ((subSize - 1) / BLOCK_SIZE + 1), BLOCK_SIZE >> >(pos, posSub, subSize);

	cudaFree(flagsDevice);
	cudaFree(switchIdx);
	cudaFree(endend);
	cudaFree(nextSwitch);
	free(blockNumArray);
}