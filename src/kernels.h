#ifdef __cplusplus
extern "C" {
    #endif

    void dark_channel_run(float * d_image, float * d_darkc, float * d_index, int N, int H, int W, int wsize);
    void dark_extract_run(float * d_image, float * d_index, float * d_darkc, int N, int H, int W);
    void place_back_run(float * d_darkc, float * d_index, float * d_image, float * d_accum, int N, int H, int W, int wsize);

    #ifdef __cplusplus
}
#endif
