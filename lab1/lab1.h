#pragma once
//#define Debug

#include "SyncedMemory.h"
#include <cstdint>
#include <cmath>
#include <memory>
#include <fstream>

using std::unique_ptr;

struct Lab1VideoInfo 
{
	unsigned w, h, n_frame;
	unsigned fps_n, fps_d;
};

class Lab1VideoGenerator 
{
	struct Impl;
	unique_ptr<Impl> impl;
public:
	Lab1VideoGenerator();
	~Lab1VideoGenerator();
	void get_info(Lab1VideoInfo &info);
	void Generate(uint8_t *yuv);
	uint8_t *RGB2YUV(uint8_t *rgb);

	void Diffuse(int NX, int NY, int b, float mvx, float mvy
		, float *x, float *x0, float diff, float dt);
	void Project(int NX, int NY, float mvx, float mvy
		, float *vx, float *vy, float *p, float *div);

	void UpdateForce();

	void UpdateDensityRGB();

	void UpdateDensity();

	void DisplayRGB(uint8_t *data);

	void Display(uint8_t *data);
	
	void InitSource();

	char *h_bd, *d_bd;

	float *h_sd, *d_sd;
	float *h_sdr, *d_sdr;
	float *h_sdg, *d_sdg;
	float *h_sdb, *d_sdb;
	float *h_svx, *d_svx;
	float *h_svy, *d_svy;

	float *h_test;

	float *d_vx;
	float *d_vx0;
	float *d_vy;
	float *d_vy0;
	float *d_d;
	float *d_d0;
	float *d_dr, *d_dg, *d_db;
	float *d_dr0, *d_dg0, *d_db0;
	float *h_dInit, *h_drInit, *h_dgInit, *h_dbInit;
	float *h_vxInit, *h_vyInit;

	float *d_p, *d_div;
};
