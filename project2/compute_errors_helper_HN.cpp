#include <iostream>
#include <math.h>
#include "mex.h"
using namespace std;

#define N_IMAGES 51852


double error;
double theta;
double s;


void compute_errors_helper(double feature_single[N_IMAGES], int s_pool[2], double theta_pool[40], double w[N_IMAGES], int Y[N_IMAGES]){
    
    double error_min = N_IMAGES + 1;
    double s_cur;
    double theta_cur;
    double feature_cur;
    int    y_filter[N_IMAGES];
    double error_cur;


    for(int i=0; i<2; i++){

        s_cur = s_pool[i];

        for(int j=0; j<40; j++){
        
            theta_cur = theta_pool[j];
            error_cur = 0;

            for(int k=0; k<N_IMAGES; k++){

                feature_cur = feature_single[k];
                if(feature_cur < theta_cur){
                    y_filter[k] = -s_cur;
                }else{
                    y_filter[k] = s_cur;
                }

                if(y_filter[k] != Y[k]){
                    error_cur = error_cur + w[k];
                }

            }

            if(error_cur<error_min){
                error_min = error_cur;

                error = error_cur;
                theta = theta_cur;
                s = s_cur;
            }

        }

    }
    
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]) {
    
    double in_0[N_IMAGES];
    int    in_1[2];
    double in_2[40];
    double in_3[N_IMAGES];
    int    in_4[N_IMAGES];


    for(int i=0; i<N_IMAGES; i++){
        in_0[i] = double((mxGetPr(prhs[0]))[i]);
        in_3[i] = double((mxGetPr(prhs[3]))[i]);
        in_4[i] = int((mxGetPr(prhs[4]))[i]);
    }

    for(int i=0; i<2; i++){
        in_1[i] = int((mxGetPr(prhs[1]))[i]);
    }

    for(int i=0; i<40; i++){
        in_2[i] = double((mxGetPr(prhs[2]))[i]);
    }


    double *out_0;
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    out_0 = mxGetPr(plhs[0]);

    double *out_1;
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    out_1 = mxGetPr(plhs[1]);

    double *out_2;
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    out_2 = mxGetPr(plhs[2]);


	compute_errors_helper(in_0, in_1, in_2, in_3, in_4);


    out_0[0] = error;
    out_1[0] = theta;
    out_2[0] = s;

    
}
