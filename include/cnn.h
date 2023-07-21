#ifndef CNN_H
#define CNN_H

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv4/opencv2/opencv.hpp>

#define MAX_PATH_LENGTH 256
#define STD (255 * 0.5)  // 0.5
#define MEAN (255 * 0.5) // 0.5

//-------------------------------------------------------------MACROS/GLOBALS---------------------------------------------------//

#define MAX_LINE_SIZE 2000000
#define DECIMAL_PLACE_FACTOR 1000

#define COMPUTE_OUTPUT_SIZE(N_in, padding, filter_size, stride) (N_in + 2 * padding - filter_size) / stride + 1

inline float relu(float x) {
    return (x > 0) ? x : 0;
}

float relu(float x);

//------------------------------------------------------------Lenet-Configuration------------------------------//

// Layer 1 - convolution + tanh
#define INPUT_FILTERS_1 3
#define INPUT_ROWS_1 128
#define INPUT_COLS_1 128
#define KERNEL_SIZE_1 3
#define STRIDE_1 2
#define PADDING_1 1
#define NUM_FILTERS_1 16

// Layer 2 - Max pooling
#define INPUT_FILTERS_2 NUM_FILTERS_1
#define INPUT_ROWS_2 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_1, PADDING_1, KERNEL_SIZE_1, STRIDE_1)
#define INPUT_COLS_2 COMPUTE_OUTPUT_SIZE(INPUT_COLS_1, PADDING_1, KERNEL_SIZE_1, STRIDE_1)
#define KERNEL_SIZE_2 2
#define STRIDE_2 1
#define PADDING_2 0
#define NUM_FILTERS_2 32

typedef struct Params {
    float weights1[NUM_FILTERS_1][INPUT_FILTERS_1][KERNEL_SIZE_1][KERNEL_SIZE_1];
    float biases1[NUM_FILTERS_1];
} Params;

typedef struct ImageData {
    // float image[MAX_IMAGE_HEIGHT][MAX_IMAGE_WIDTH][MAX_IMAGE_CHANNELS][MAX_IMAGE_FILTERS];
    float image[INPUT_FILTERS_1][INPUT_ROWS_1 + PADDING_1][INPUT_COLS_1 + PADDING_1];
    float layer_1[INPUT_FILTERS_2][INPUT_ROWS_2][INPUT_COLS_2];
    int low_width;
    int low_height;
    int filters;
    int width;
    int height;
} ImageData;

int forwardPass(ImageData& inputData, Params& param);

void layer_1_conv(ImageData& inputData, Params& param, int padding, int further_padding, int stride, int kernel_size, int in_filters, int out_filters);

void layer_2_conv(ImageData& inputData, Params& param, int padding, int further_padding, int stride, int kernel_size, int in_filters, int out_filters);

void loadDataset(const char* folderPath, ImageData& imageData, int& test_set_size, int padding, Params& param, cv::Mat& image);

void loadParams(const char* paramPath, Params& params);

#endif // CNN_H
