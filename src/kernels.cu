#define Inf 9999999

void __global__ dark_channel_ker(float * image, float * darkc, float * index, int N, int H, int W, int wsize)
{
    int C = 3;
    int D = 1;
    int hsize = (wsize - 1) / 2;

    int ids = blockDim.x * blockIdx.x + threadIdx.x;
    int idt = gridDim.x * blockDim.x;

    for(; ids<W*H*N; ids+=idt)
    {
        // ids = W*H*n + W*h + w
        int idw = ids % W;
        ids = (ids - idw) / W;
        int idh = ids % H;
        ids = (ids - idh) / H;
        int idn = ids;

        float tmpc = Inf;
        float tmp1 = 0;
        float tmp2 = 0;
        float tmp3 = 0;

        for(int idc=0; idc<C; idc++)
        {
            for(int p=idh-hsize; p<idh+hsize+1; p++)
            {
                for(int q=idw-hsize; q<idw+hsize+1; q++)
                {
                    if(p>-1 && p<H && q>-1 && q<W)
                    {
                        float pixel_value = image[W*H*C*idn + W*H*idc + W*p + q];
                        if(pixel_value<tmpc)
                        {
                            tmpc = pixel_value;
                            tmp1 = idc;
                            tmp2 = p;
                            tmp3 = q;
                        }
                    }
                }
            }
        }

        darkc[W*H*D*idn + W*H*0 + W*idh + idw] = tmpc;
        index[W*H*C*idn + W*H*0 + W*idh + idw] = tmp1;
        index[W*H*C*idn + W*H*1 + W*idh + idw] = tmp2;
        index[W*H*C*idn + W*H*2 + W*idh + idw] = tmp3;
    }
}

void __global__ dark_extract_ker(float * image, float * index, float * darkc, int N, int H, int W)
{
    int C = 3;

    int ids = blockDim.x * blockIdx.x + threadIdx.x;
    int idt = gridDim.x * blockDim.x;

    for(; ids<N*H*W; ids+=idt)
    {
        int idw = ids % W;
        ids = (ids - idw) / W;
        int idh = ids % H;
        ids = (ids - idh) / H;
        int idn = ids;

        int idz = index[W*H*C*idn + W*H*0 + W*idh + idw];
        int idx = index[W*H*C*idn + W*H*1 + W*idh + idw];
        int idy = index[W*H*C*idn + W*H*2 + W*idh + idw];

        darkc[W*H*idn + W*idh + idw] = image[W*H*C*idn + W*H*idz + W*idx + idy];
    }
}

void __global__ place_back_ker(float * darkc, float * index, float * image, float * accum, int N, int H, int W, int wsize)
{
    int C = 3;
    int hsize = (wsize - 1) / 2;

    int ids = blockDim.x * blockIdx.x + threadIdx.x;
    int idt = gridDim.x * blockDim.x;

    for(; ids<N*C*H*W; ids+=idt)
    {
        int idw = ids % W;
        ids = (ids - idw) / W;
        int idh = ids % H;
        ids = (ids - idh) / H;
        int idc = ids % C;
        ids = (ids - idc) / C;
        int idn = ids;

        image[W*H*C*idn + W*H*idc + W*idh + idw] = 0;
        accum[W*H*C*idn + W*H*idc + W*idh + idw] = 0;
        float tmpi = 0;
        float tmpa = 0;

        for(int p=idh-hsize; p<idh+hsize+1; p++)
        {
            for(int q=idw-hsize; q<idw+hsize+1; q++)
            {
                if(p>-1 && p<H && q>-1 && q<W)
                {
                    int idz = (int) index[W*H*C*idn + W*H*0 + W*p + q];
                    int idx = (int) index[W*H*C*idn + W*H*1 + W*p + q];
                    int idy = (int) index[W*H*C*idn + W*H*2 + W*p + q];
                    if(idx==idh && idy==idw && idz==idc)
                    {
                        tmpi += darkc[W*H*idn + W*p + q];
                        tmpa += 1;
                    }
                }
            }
        }
        image[W*H*C*idn + W*H*idc + W*idh + idw] = tmpi;
        accum[W*H*C*idn + W*H*idc + W*idh + idw] = tmpa;
    }
}

#ifdef __cplusplus
extern "C"
{
    #endif

    void dark_channel_run(float * d_image, float * d_darkc, float * d_index, int N, int H, int W, int wsize)
    {
        int const threadsPerBlock = 1024;
        int const blocksPerGrid = (N*H*W + threadsPerBlock - 1) / threadsPerBlock;
        dark_channel_ker<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_darkc, d_index, N, H, W, wsize);
    }

    void dark_extract_run(float * d_image, float * d_index, float * d_darkc, int N, int H, int W)
    {
        int const threadsPerBlock = 1024;
        int const blocksPerGrid = (N*H*W + threadsPerBlock - 1) / threadsPerBlock;
        dark_extract_ker<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_index, d_darkc, N, H, W);
    }

    void place_back_run(float * d_darkc, float * d_index, float * d_image, float * d_accum, int N, int H, int W, int wsize)
    {
        int const threadsPerBlock = 1024;
        int const blocksPerGrid = (N*3*H*W + threadsPerBlock - 1) / threadsPerBlock;
        place_back_ker<<<blocksPerGrid, threadsPerBlock>>>(d_darkc, d_index, d_image, d_accum, N, H, W, wsize);
    }

    #ifdef __cplusplus
}
#endif
