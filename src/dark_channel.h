void dark_channel(THCudaTensor * image, THCudaTensor * darkc, THCudaTensor * index, int wsize);

void dark_extract(THCudaTensor * image, THCudaTensor * index, THCudaTensor * darkc);

void place_back(THCudaTensor * darkc, THCudaTensor * index, THCudaTensor * image, THCudaTensor * accum, int wsize);
