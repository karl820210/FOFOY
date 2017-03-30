#include "lab1.h"

#ifdef Debug
#include "cv.h"
#include "highgui.h"
#endif

static const int W = 640;
static const int H = 640;
static const unsigned NFRAME = 1200;
static const unsigned FPS = 24;

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
#define DIFF 0.1
#define evoSpeed 6
#define PI 3.14159265

static const float edt = evoSpeed * 1.0 / (float)FPS;

int _1D(int x, int y, int NX, int NY)
{
	int idx = x + (NX * y);
	if (x < 0)
		idx += NX;
	if (x >= NX)
		idx -= NX;
	if (y < 0)
		idx += (NX * (NY - 1));
	if (y >= NY)
		idx -= (NX * (NY - 1));
	return idx;
	//return x + (NX * y); 
}
__device__ int _D1D(int x, int y, int NX, int NY)
{ 
	int idx = x + (NX * y);
	if (x < 0)
		idx += NX;
	if (x >= NX)
		idx -= NX;
	if (y < 0)
		idx += (NX * (NY - 1));
	if (y >= NY)
		idx -= (NX * (NY - 1));
	return idx;
	//return x + (NX * y); 
}

struct Lab1VideoGenerator::Impl
{
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl)
{
	int mSize = W * H;
	h_bd = (char*)calloc(mSize, sizeof(char));
	h_sd = (float*)calloc(mSize, sizeof(float));
	h_sdr = (float*)calloc(mSize, sizeof(float));
	h_sdg = (float*)calloc(mSize, sizeof(float));
	h_sdb = (float*)calloc(mSize, sizeof(float));
	h_dInit = (float*)calloc(mSize, sizeof(float));
	h_drInit = (float*)calloc(mSize, sizeof(float));
	h_dgInit = (float*)calloc(mSize, sizeof(float));
	h_dbInit = (float*)calloc(mSize, sizeof(float));
	h_svx = (float*)calloc(mSize, sizeof(float));
	h_svy = (float*)calloc(mSize, sizeof(float));
	h_vxInit = (float*)calloc(mSize, sizeof(float));
	h_vyInit = (float*)calloc(mSize, sizeof(float));
	h_test = (float*)calloc(mSize, sizeof(float));


	InitSource();
	

	cudaMalloc((void **)&d_bd, sizeof(char) * mSize);
	cudaMalloc((void **)&d_sd, sizeof(float) * mSize);
	cudaMalloc((void **)&d_sdr, sizeof(float) * mSize);
	cudaMalloc((void **)&d_sdg, sizeof(float) * mSize);
	cudaMalloc((void **)&d_sdb, sizeof(float) * mSize);
	cudaMalloc((void **)&d_svx, sizeof(float) * mSize);
	cudaMalloc((void **)&d_svy, sizeof(float) * mSize);

	cudaMalloc((void **)&d_vx, sizeof(float) * mSize);
	cudaMalloc((void **)&d_vx0, sizeof(float) * mSize);
	cudaMalloc((void **)&d_vy, sizeof(float) * mSize);
	cudaMalloc((void **)&d_vy0, sizeof(float) * mSize);
	cudaMalloc((void **)&d_d, sizeof(float) * mSize);
	cudaMalloc((void **)&d_d0, sizeof(float) * mSize);
	cudaMalloc((void **)&d_dr, sizeof(float) * mSize);
	cudaMalloc((void **)&d_dr0, sizeof(float) * mSize);
	cudaMalloc((void **)&d_dg, sizeof(float) * mSize);
	cudaMalloc((void **)&d_dg0, sizeof(float) * mSize);
	cudaMalloc((void **)&d_db, sizeof(float) * mSize);
	cudaMalloc((void **)&d_db0, sizeof(float) * mSize);

	cudaMalloc((void **)&d_p, sizeof(float) * mSize);
	cudaMalloc((void **)&d_div, sizeof(float) * mSize);

	cudaMemcpy(d_bd, h_bd, sizeof(char) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sd, h_sd, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sdr, h_sdr, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sdg, h_sdg, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sdb, h_sdb, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d0, h_dInit, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dr0, h_drInit, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dg0, h_dgInit, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_db0, h_dbInit, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_svx, h_svx, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_svy, h_svy, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vx0, h_vxInit, sizeof(float) * mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy0, h_vyInit, sizeof(float) * mSize, cudaMemcpyHostToDevice);

	cudaMemset(d_vx, 0, W*H);
	cudaMemset(d_vy, 0, W*H);

	cudaMemset(d_d, 0, W*H);
	cudaMemset(d_dr, 0, W*H);
	cudaMemset(d_dg, 0, W*H);
	cudaMemset(d_db, 0, W*H);
	//cudaMemcpy(h_data, d_data, size,cudaMemcpyDeviceToHost);

}

Lab1VideoGenerator::~Lab1VideoGenerator()
{
	/*free(h_bd);
	free(h_s);*/
}

void Lab1VideoGenerator::InitSource()
{
	for (int x = 0; x < W; x++)
	{
		for (int y = 0; y < H; y++)
		{
			/*
			if (x == 0 || x == W - 1 || y == 0 || y == H - 1)
			h_bd[_1D(x, y, W, H)] = 'B';
			else
			h_bd[_1D(x, y, W, H)] = '_';
			*/
			/*
			int shW = 80;
			int shH = 40;

			if (x > W / 2 - shH && x < W / 2 + shH && y > H / 2 - shW && y < H / 2 + shW)
			{
			h_sd[_1D(x - 100, y, W, H)] = 20.0;
			h_sd[_1D(x + 100, y, W, H)] = 20.0;
			}
			if (x > W / 2 - shW && x < W / 2 + shW && y > H / 2 - shH && y < H / 2 + shH)
			{
			h_sd[_1D(x, y - 100, W, H)] = -20.0;
			h_sd[_1D(x, y + 100, W, H)] = -20.0;
			}
			*/

			float r = sqrt((x - W / 2) * (x - W / 2) + (y - H / 2) * (y - H / 2));

			float sr1 = 70, sr2 = 110;
			if (r >= sr1 && r <= sr2)
			{
				float degree = atan2(y - H / 2, x - W / 2) * 180 / PI;
				float shD = 45;
				if ((degree >= 90 - shD && degree <= 90 + shD) || (degree >= -90 - shD && degree <= -90 + shD))
					h_sd[_1D(x, y, W, H)] = -5.0;
				if ((degree >= 0 - shD && degree <= 0 + shD) || (degree >= 180 - shD || degree <= -180 + shD))
					h_sd[_1D(x, y, W, H)] = 5.0;
			}

			sr1 = 120, sr2 = 200;
			if (r >= sr1 && r <= sr2)
			{
				float degree = atan2(y - H / 2, x - W / 2) * 180 / PI;
				float shD = 40;
				if ((degree >= 90 - shD && degree <= 90 + shD) || (degree >= -90 - shD && degree <= -90 + shD))
					h_sd[_1D(x, y, W, H)] = 10.0;
				if ((degree >= 0 - shD && degree <= 0 + shD) || (degree >= 180 - shD || degree <= -180 + shD))
					h_sd[_1D(x, y, W, H)] = -10.0;
			}


			float r1 = 100, r2 = 200;
			float rv2 = -95 * PI / 180;
			if (r > r1 && r <= r2)
			{
				h_svx[_1D(x - 123, y + 46, W, H)] = ((W / 2 - x) * cos(rv2) - (H / 2 - y) * sin(rv2)) / r2 * 0.005;
				h_svy[_1D(x - 123, y + 46, W, H)] = ((W / 2 - x) * sin(rv2) + (H / 2 - y) * cos(rv2)) / r2 * 0.005;
			}
			if (r > r1 && r <= r2)
			{
				h_svx[_1D(x + 123, y - 46, W, H)] = ((W / 2 - x) * cos(rv2) - (H / 2 - y) * sin(rv2)) / r2 * 0.005;
				h_svy[_1D(x + 123, y - 46, W, H)] = ((W / 2 - x) * sin(rv2) + (H / 2 - y) * cos(rv2)) / r2 * 0.005;
			}
		}
	}
}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info)
{
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = FPS;
	info.fps_d = 1;
};

__device__ void SetBnd(int NX, int NY, float mvx, float mvy, float *x, int i, int j, int b, char *bd)
{
	//simulation external boundaries
	/*
	if (i == 0)
		x[_D1D(i, j, NX, NY)] = (b == 1) ? -x[_D1D(i + 1, j, NX, NY)] : x[_D1D(i + 1, j, NX, NY)];
	if (i == NX - 1)
		x[_D1D(i, j, NX, NY)] = (b == 1) ? -x[_D1D(i - 1, j, NX, NY)] : x[_D1D(i - 1, j, NX, NY)];
	if (j == 0)
		x[_D1D(i, j, NX, NY)] = (b == 2) ? -x[_D1D(i, j + 1, NX, NY)] : x[_D1D(i, j + 1, NX, NY)];
	if (j == NY - 1)
		x[_D1D(i, j, NX, NY)] = (b == 2) ? -x[_D1D(i, j - 1, NX, NY)] : x[_D1D(i, j - 1, NX, NY)];

	//simulation corner boundaries
	if ((i == 0) && (j == 0))
		x[_D1D(i, j, NX, NY)] = 0.5f * (x[_D1D(i + 1, j, NX, NY)] + x[_D1D(i, j + 1, NX, NY)]);
	if ((i == NX - 1) && (j == 0))
		x[_D1D(i, j, NX, NY)] = 0.5f * (x[_D1D(i - 1, j, NX, NY)] + x[_D1D(i, j + 1, NX, NY)]);
	if ((i == 0) && (j == NY - 1))
		x[_D1D(i, j, NX, NY)] = 0.5f * (x[_D1D(i + 1, j, NX, NY)] + x[_D1D(i, j - 1, NX, NY)]);
	if ((i == NX - 1) && (j == NY - 1))
		x[_D1D(i, j, NX, NY)] = 0.5f * (x[_D1D(i - 1, j, NX, NY)] + x[_D1D(i, j - 1, NX, NY)]);
		*/
	//simulation internal boundaries
	if ((bd[_D1D(i, j, NX, NY)]) == 'B')
		x[_D1D(i, j, NX, NY)] = 0;

	//simulation moving boundaries
	if ((bd[_D1D(i, j, NX, NY)]) == 'M')
	{
		x[_D1D(i, j, NX, NY)] = (b == 1) ? mvx : x[_D1D(i, j, NX, NY)];
		x[_D1D(i, j, NX, NY)] = (b == 2) ? mvy : x[_D1D(i, j, NX, NY)];
		x[_D1D(i, j, NX, NY)] = (b == 0) ? 0 : x[_D1D(i, j, NX, NY)];
	}


}

__global__ void Gsrb(int NX, int NY, float mvx, float mvy,
	float *x, float *x0, float a, float iter, int b, char *bd)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	for (int it = 0; it < 10; it++)
	{
		//if ((i != 0) && (j != 0) && (i != NX - 1) && (j != NY - 1))
		{
#if 0
			if ((i + j) % 2 == 0)
			{
				x[_D1D(i, j, NX, NY)] = (x0[_D1D(i, j, NX, NY)]
					+ a * (x[_D1D(i - 1, j, NX, NY)] + x[_D1D(i + 1, j, NX, NY)]
					+ x[_D1D(i, j - 1, NX, NY)] + x[_D1D(i, j + 1, NX, NY)])) / iter;
			}
			syncthreads();
			if ((i + j) % 2 != 0)
			{
				x[_D1D(i, j, NX, NY)] = (x0[_D1D(i, j, NX, NY)]
					+ a * (x[_D1D(i - 1, j, NX, NY)] + x[_D1D(i + 1, j, NX, NY)]
					+ x[_D1D(i, j - 1, NX, NY)] + x[_D1D(i, j + 1, NX, NY)])) / iter;
			}
			syncthreads();
#else
			x[_D1D(i, j, NX, NY)] = (x0[_D1D(i, j, NX, NY)] + a * (x[_D1D(i - 1, j, NX, NY)] + x[_D1D(i + 1, j, NX, NY)]
														 + x[_D1D(i, j - 1, NX, NY)] + x[_D1D(i, j + 1, NX, NY)])) / iter;
			syncthreads();
#endif
		}
		//SetBnd(NX, NY, mvx, mvy, x, i, j, b, bd);
	}
}

__global__ void GsrbIter(int NX, int NY, float mvx, float mvy,
	float *x, float *x0, float a, float iter, int b, char *bd)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i != 0) && (j != 0) && (i != NX - 1) && (j != NY - 1))
	{
#if 1
		if ((i + j) % 2 == 0)
		{
			x[_D1D(i, j, NX, NY)] = (x0[_D1D(i, j, NX, NY)]
				+ a * (x[_D1D(i - 1, j, NX, NY)] + x[_D1D(i + 1, j, NX, NY)]
				+ x[_D1D(i, j - 1, NX, NY)] + x[_D1D(i, j + 1, NX, NY)])) / iter;
		}
		syncthreads();
		if ((i + j) % 2 != 0)
		{
			x[_D1D(i, j, NX, NY)] = (x0[_D1D(i, j, NX, NY)]
				+ a * (x[_D1D(i - 1, j, NX, NY)] + x[_D1D(i + 1, j, NX, NY)]
				+ x[_D1D(i, j - 1, NX, NY)] + x[_D1D(i, j + 1, NX, NY)])) / iter;
		}
#else
		x[_D1D(i, j, NX, NY)] = (x0[_D1D(i, j, NX, NY)]
			+ a * (x[_D1D(i - 1, j, NX, NY)] + x[_D1D(i + 1, j, NX, NY)]
			+ x[_D1D(i, j - 1, NX, NY)] + x[_D1D(i, j + 1, NX, NY)])) / iter;
		syncthreads();
#endif
	}
	//SetBnd(NX, NY, mvx, mvy, x, i, j, b, bd);
}

__global__ void AddSource(int NX, int NY, float *x, float *s, float dt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	x[_D1D(i, j, NX, NY)] += dt * s[_D1D(i, j, NX, NY)];
}

__global__ void Advect(int NX, int NY, float mvx, float mvy,
	int b, float *d, float *d0, float *vx, float *vy, float dt, char *bd)
{
	int i0, j0, i1, j1;
	float xf, yf, s0, t0, s1, t1, dtx, dty;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	/*
	dtx = dt * (NX - 2);
	dty = dt * (NY - 2);
	*/
	dtx = dt * NX;
	dty = dt * NY;
	//if ((i != 0) && (j != 0) && (i != NX - 1) && (j != NY - 1))
	{
		xf = (float)i - (dtx * vx[_D1D(i, j, NX, NY)]);
		yf = (float)j - (dty * vy[_D1D(i, j, NX, NY)]);
		/*
		if (xf < 0.5f) xf = 0.5f;
		if (xf >(NX + 0.5f)) xf = NX + 0.5f;
		i0 = (int)xf; i1 = i0 + 1;

		if (yf < 0.5f) yf = 0.5f;
		if (yf >(NY + 0.5f)) yf = NY + 0.5f;
		j0 = (int)yf; j1 = j0 + 1;
		*/
		if (xf < 0) xf += NX;
		if (xf >= NX) xf -= NX;
		i0 = (int)xf; i1 = i0 + 1;

		if (yf < 0) yf += NY;
		if (yf > NY) yf -= NY;
		j0 = (int)yf; j1 = j0 + 1;
		s1 = xf - i0; s0 = 1 - s1;
		t1 = yf - j0; t0 = 1 - t1;

		d[_D1D(i, j, NX, NY)] = s0 * (t0 * d0[_D1D(i0, j0, NX, NY)] + t1 * d0[_D1D(i0, j1, NX, NY)])
							+ s1 * (t0 * d0[_D1D(i1, j0, NX, NY)] + t1 * d0[_D1D(i1, j1, NX, NY)]);
	}
	syncthreads();
	//SetBnd(NX, NY, mvx, mvy, d, i, j, b, bd);
}

void Lab1VideoGenerator::Diffuse(int NX, int NY, int b, float mvx, float mvy
	, float *x, float *x0, float diff, float dt)
{
	dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(NX / BLOCK_DIM_X, NY / BLOCK_DIM_Y);

	float a = dt * diff * NX * NY;

	Gsrb <<< grid, threads >>> (NX, NY, mvx, mvy, x, x0, a, 1 + 4 * a, b, d_bd);
	/*
	for (int it = 0; it < 4; it++)
		GsrbIter << < grid, threads >> > (NX, NY, mvx, mvy, x, x0, a, 1 + 4 * a, b, d_bd);*/
}

__global__ void Project1(int NX, int NY, float mvx, float mvy,
	float *vx, float *vy, float *p, float *div, char *bd)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	//if ((i != 0) && (j != 0) && (i != NX - 1) && (j != NY - 1))
	{
		div[_D1D(i, j, NX, NY)] = -0.5f * ((vx[_D1D(i + 1, j, NX, NY)] - vx[_D1D(i - 1, j, NX, NY)]) / NX
			+ (vy[_D1D(i, j + 1, NX, NY)] - vy[_D1D(i, j - 1, NX, NY)]) / NY);
		p[_D1D(i, j, NX, NY)] = 0;
	}
	syncthreads();
	//SetBnd(NX, NY, mvx, mvy, div, i, j, 0, bd);
	//SetBnd(NX, NY, mvx, mvy, p, i, j, 0, bd);
}

__global__ void Project2(int NX, int NY, float mvx, float mvy
	, float *vx, float *vy, float *p, char *bd)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	//if ((i != 0) && (j != 0) && (i != NX - 1) && (j != NY - 1))
	{
		vx[_D1D(i, j, NX, NY)] -= 0.5f * NX * (p[_D1D(i + 1, j, NX, NY)] - p[_D1D(i - 1, j, NX, NY)]);
		vy[_D1D(i, j, NX, NY)] -= 0.5f * NY * (p[_D1D(i, j + 1, NX, NY)] - p[_D1D(i, j - 1, NX, NY)]);
	}
	syncthreads();
	//SetBnd(NX, NY, mvx, mvy, vx, i, j, 1, bd);
	//SetBnd(NX, NY, mvx, mvy, vy, i, j, 2, bd);
}

void Lab1VideoGenerator::Project(int NX, int NY, float mvx, float mvy
	, float *vx, float *vy, float *p, float *div)
{
	dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(NX / BLOCK_DIM_X, NY / BLOCK_DIM_Y);

	Project1 <<< grid, threads >>> (NX, NY, mvx, mvy, vx, vy, p, div, d_bd);
	Gsrb << < grid, threads >> > (NX, NY, mvx, mvy, p, div, 1.0f, 4.0f, 0, d_bd);
	Project2 <<< grid, threads >>> (NX, NY, mvx, mvy, vx, vy, p, d_bd);
}

__global__ void CopyDataRGB(int NX, int NY, uint8_t *data, float *dr, float *dr0, float *dg, float *dg0, float *db, float *db0, float *vx, float *vx0, float *vy, float *vy0)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	data[(j * NX + i) * 3] = (uint8_t)(dr[_D1D(i, j, NX, NY)] > 255) ? 255 : dr[_D1D(i, j, NX, NY)];
	data[(j * NX + i) * 3 + 1] = (uint8_t)(dg[_D1D(i, j, NX, NY)] > 255) ? 255 : dg[_D1D(i, j, NX, NY)];
	data[(j * NX + i) * 3 + 2] = (uint8_t)(db[_D1D(i, j, NX, NY)] > 255) ? 255 : db[_D1D(i, j, NX, NY)];

	dr0[_D1D(i, j, NX, NY)] = dr[_D1D(i, j, NX, NY)];
	dg0[_D1D(i, j, NX, NY)] = dg[_D1D(i, j, NX, NY)];
	db0[_D1D(i, j, NX, NY)] = db[_D1D(i, j, NX, NY)];
	vx0[_D1D(i, j, NX, NY)] = vx[_D1D(i, j, NX, NY)];
	vy0[_D1D(i, j, NX, NY)] = vy[_D1D(i, j, NX, NY)];
}

__global__ void CopyData(int NX, int NY, uint8_t *data, float *d, float *d0, float *vx, float *vx0, float *vy, float *vy0, char *bd)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	/*
	if (d[_D1D(i, j, NX, NY)] > 0)
		data[(j * NX + i) * 3] = (uint8_t)(d[_D1D(i, j, NX, NY)] > 255) ? 255 : d[_D1D(i, j, NX, NY)];
	else
		data[(j * NX + i) * 3 + 2] = (uint8_t)(d[_D1D(i, j, NX, NY)] < -255) ? 255 : -d[_D1D(i, j, NX, NY)];
	data[(j * NX + i) * 3 + 1] = (bd[_D1D(i, j, NX, NY)] == 'B') ? 255 : 0;
	*/
	float R, G, B;
	R = G = B = 0;
	if (d[_D1D(i, j, NX, NY)] > 0)
		R = (d[_D1D(i, j, NX, NY)] > 255) ? 255 : d[_D1D(i, j, NX, NY)];
	else
		B = (d[_D1D(i, j, NX, NY)] < -255) ? 255 : -d[_D1D(i, j, NX, NY)];
	float Y = 0.299 * R + 0.587 * G + 0.114 * B;
	float U = -0.169 * R + -0.331 * G + 0.5 * B + 128;
	float V = 0.5 * R + -0.419 * G + -0.081 * B + 128;

	data[i * NX + j] = (Y > 255) ? 255 : (Y < 0) ? 0 : Y;
	data[(int)(NX * NY + (i / 2)*(NX / 2) + j / 2)] = (U > 255) ? 255 : (U < 0) ? 0 : U;
	data[(int)(NX * NY * 1.25 + (i / 2)*(NX / 2) + j / 2)] = (V > 255) ? 255 : (V < 0) ? 0 : V;

	d0[_D1D(i, j, NX, NY)] = d[_D1D(i, j, NX, NY)];
	vx0[_D1D(i, j, NX, NY)] = vx[_D1D(i, j, NX, NY)];
	vy0[_D1D(i, j, NX, NY)] = vy[_D1D(i, j, NX, NY)];
}

void Lab1VideoGenerator::UpdateForce()
{
	dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(W / BLOCK_DIM_X, H / BLOCK_DIM_Y);

	AddSource << < grid, threads >> > (W, H, d_vx0, d_svx, edt);
	AddSource << < grid, threads >> > (W, H, d_vy0, d_svy, edt);

	Advect << < grid, threads >> > (W, H, 1, 0, 0, d_vx, d_vx0, d_vx, d_vy, edt, d_bd);
	Advect << < grid, threads >> > (W, H, 2, 0, 0, d_vy, d_vy0, d_vx, d_vy, edt, d_bd);

	Diffuse(W, H, 1, 0, 0, d_vx, d_vx0, DIFF, edt);
	Diffuse(W, H, 2, 0, 0, d_vy, d_vy0, DIFF, edt);

	Project(W, H, 0, 0, d_vx, d_vy, d_p, d_div);
}

void Lab1VideoGenerator::UpdateDensityRGB()
{
	dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(W / BLOCK_DIM_X, H / BLOCK_DIM_Y);

	AddSource << < grid, threads >> > (W, H, d_dr0, d_sdr, edt);
	AddSource << < grid, threads >> > (W, H, d_dg0, d_sdg, edt);
	AddSource << < grid, threads >> > (W, H, d_db0, d_sdb, edt);

	Advect << < grid, threads >> > (W, H, 0, 0, 0, d_dr, d_dr0, d_vx, d_vy, edt, d_bd);
	Advect << < grid, threads >> > (W, H, 0, 0, 0, d_dg, d_dg0, d_vx, d_vy, edt, d_bd);
	Advect << < grid, threads >> > (W, H, 0, 0, 0, d_db, d_db0, d_vx, d_vy, edt, d_bd);

	Diffuse(W, H, 0, 0, 0, d_dr, d_dr0, DIFF, edt);
	Diffuse(W, H, 0, 0, 0, d_dg, d_dg0, DIFF, edt);
	Diffuse(W, H, 0, 0, 0, d_db, d_db0, DIFF, edt);
}

void Lab1VideoGenerator::UpdateDensity()
{
	dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(W / BLOCK_DIM_X, H / BLOCK_DIM_Y);

	AddSource << < grid, threads >> > (W, H, d_d0, d_sd, edt);

	Advect << < grid, threads >> > (W, H, 0, 0, 0, d_d, d_d0, d_vx, d_vy, edt, d_bd);

	Diffuse(W, H, 0, 0, 0, d_d, d_d0, DIFF, edt);
}

void Lab1VideoGenerator::DisplayRGB(uint8_t *data)
{
	dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(W / BLOCK_DIM_X, H / BLOCK_DIM_Y);

	CopyDataRGB << < grid, threads >> > (W, H, data, d_dr, d_dr0, d_dg, d_dg0, d_db, d_db0, d_vx, d_vx0, d_vy, d_vy0);
}

void Lab1VideoGenerator::Display(uint8_t *data)
{
	dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(W / BLOCK_DIM_X, H / BLOCK_DIM_Y);

	CopyData << < grid, threads >> > (W, H, data, d_d, d_d0, d_vx, d_vx0, d_vy, d_vy0, d_bd);
}

void Lab1VideoGenerator::Generate(uint8_t *yuv)
{
	UpdateForce();
	UpdateDensity();
	Display(yuv);
	int mSize = W * H;
#ifdef Debug
	//uint8_t *h_yuv = (uint8_t*)malloc(sizeof(uint8_t)*mSize * 3);
	//cudaMemcpy(h_yuv, yuv, sizeof(uint8_t) * mSize * 3, cudaMemcpyDeviceToHost);
#if 0
	cudaMemcpy(h_test, d_vx0, sizeof(float) * mSize, cudaMemcpyDeviceToHost);
	cv::Mat matDr(H, W, CV_32FC1, h_test);
	cv::imshow("vx", matDr);
	cudaMemcpy(h_test, d_vy, sizeof(float) * mSize, cudaMemcpyDeviceToHost);
	cv::Mat matDg(H, W, CV_32FC1, h_test);
	cv::imshow("vy", matDg);
#endif
	//cv::Mat matHyuv(H, W, CV_8UC3, h_yuv);
	//cv::imshow("h_yuv", matHyuv);
#endif
	//++(impl->t);
}
