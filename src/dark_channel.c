#include "kernels.h"
#include "THC/THC.h"

extern THCState *state;

void dark_channel(THCudaTensor * image, THCudaTensor * darkc, THCudaTensor * index, int wsize)
{
    float * d_image = (float *)THCudaTensor_data(state, image);
    float * d_darkc = (float *)THCudaTensor_data(state, darkc);
    float * d_index = (float *)THCudaTensor_data(state, index);
    int N = image->size[0];
    int H = image->size[2];
    int W = image->size[3];
    dark_channel_run(d_image, d_darkc, d_index, N, H, W, wsize);
}

void dark_extract(THCudaTensor * image, THCudaTensor * index, THCudaTensor * darkc)
{
    float * d_image = (float *)THCudaTensor_data(state, image);
    float * d_index = (float *)THCudaTensor_data(state, index);
    float * d_darkc = (float *)THCudaTensor_data(state, darkc);
    int N = image->size[0];
    int H = image->size[2];
    int W = image->size[3];
    dark_extract_run(d_image, d_index, d_darkc, N, H, W);
}

void place_back(THCudaTensor * darkc, THCudaTensor * index, THCudaTensor * image, THCudaTensor * accum, int wsize)
{
    float * d_darkc = (float *)THCudaTensor_data(state, darkc);
    float * d_index = (float *)THCudaTensor_data(state, index);
    float * d_image = (float *)THCudaTensor_data(state, image);
    float * d_accum = (float *)THCudaTensor_data(state, accum);
    int N = image->size[0];
    int H = image->size[2];
    int W = image->size[3];
    place_back_run(d_darkc, d_index, d_image, d_accum, N, H, W, wsize);
}
