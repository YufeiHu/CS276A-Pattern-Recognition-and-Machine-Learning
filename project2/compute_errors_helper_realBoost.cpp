#include <iostream>
#include <math.h>
#include "mex.h"
using namespace std;

#define N_IMAGES 47194
#define B        30


double p[B];
double q[B];
double z;


void compute_errors_helper_realBoost(double feature_single[N_IMAGES], double h_pool[B], double w[N_IMAGES], int Y[N_IMAGES]){
    
	int bin_index;
	int j;
	double feature_cur;


	for(int i=0; i<B; i++){
		p[i] = 0;
		q[i] = 0;
	}


	for(int i=0; i<N_IMAGES; i++){

		feature_cur = feature_single[i];

		for(j=0; j<B; j++){
			if ((feature_cur>h_pool[j]) && (feature_cur<h_pool[j+1])){
				break;
			}
		}
		bin_index = j;

		if(Y[i]==1){
			p[bin_index] += w[i];
		}else{
			q[bin_index] += w[i];
		}

	}

	z = 0;
	for(int i=0; i<B; i++){
		z += sqrt(p[i] * q[i]);
	}
	z *= 2;
    
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]) {
    
    double in_0[N_IMAGES];
    double in_1[B];
    double in_2[N_IMAGES];
    int    in_3[N_IMAGES];


    for(int i=0; i<N_IMAGES; i++){
        in_0[i] = double((mxGetPr(prhs[0]))[i]);
        in_2[i] = double((mxGetPr(prhs[2]))[i]);
        in_3[i] = int((mxGetPr(prhs[3]))[i]);
    }

    for(int i=0; i<B; i++){
        in_1[i] = double((mxGetPr(prhs[1]))[i]);
    }


    double *out_0;
    plhs[0] = mxCreateDoubleMatrix(B, 1, mxREAL);
    out_0 = mxGetPr(plhs[0]);

    double *out_1;
    plhs[1] = mxCreateDoubleMatrix(B, 1, mxREAL);
    out_1 = mxGetPr(plhs[1]);

    double *out_2;
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    out_2 = mxGetPr(plhs[2]);


	compute_errors_helper_realBoost(in_0, in_1, in_2, in_3);


	for(int i=0; i<B; i++){
        out_0[i] = p[i];
        out_1[i] = q[i];
    }
    out_2[0] = z;

    
}
