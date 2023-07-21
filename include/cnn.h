#ifndef CNN_H
#define CNN_H

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv4/opencv2/opencv.hpp>

#define MAX_PATH_LENGTH 256
const float STD = (255 * 0.5f);  // 0.5
const float MEAN = (255 * 0.5f); // 0.5

//-------------------------------------------------------------MACROS/GLOBALS---------------------------------------------------//

#define MAX_LINE_SIZE 2000000
#define DECIMAL_PLACE_FACTOR 1000

#define COMPUTE_OUTPUT_SIZE(N_in, padding, filter_size, stride) (N_in + 2 * padding - filter_size) / stride + 1

inline float relu(float x) {
    return (x > 0) ? x : 0;
}

float relu(float x);

//------------------------------------------------------------Lenet-Configuration------------------------------//

// Layer 1 - convolution + relu
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
#define STRIDE_2 2
#define PADDING_2 0
#define NUM_FILTERS_2 INPUT_FILTERS_2

// Layer 3 - convolution + relu
#define INPUT_FILTERS_3 NUM_FILTERS_2
#define INPUT_ROWS_3 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_2, PADDING_2, KERNEL_SIZE_2, STRIDE_2)
#define INPUT_COLS_3 COMPUTE_OUTPUT_SIZE(INPUT_COLS_2, PADDING_2, KERNEL_SIZE_2, STRIDE_2)
#define KERNEL_SIZE_3 3
#define STRIDE_3 2
#define PADDING_3 1
#define NUM_FILTERS_3 32

// Layer 4 - Max pooling
#define INPUT_FILTERS_4 NUM_FILTERS_3
#define INPUT_ROWS_4 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_3, PADDING_3, KERNEL_SIZE_3, STRIDE_3)
#define INPUT_COLS_4 COMPUTE_OUTPUT_SIZE(INPUT_COLS_3, PADDING_3, KERNEL_SIZE_3, STRIDE_3)
#define KERNEL_SIZE_4 2
#define STRIDE_4 2
#define PADDING_4 0
#define NUM_FILTERS_4 INPUT_FILTERS_4

// Layer 5 - convolution + relu
#define INPUT_FILTERS_5 NUM_FILTERS_4
#define INPUT_ROWS_5 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_4, PADDING_4, KERNEL_SIZE_4, STRIDE_4)
#define INPUT_COLS_5 COMPUTE_OUTPUT_SIZE(INPUT_COLS_4, PADDING_4, KERNEL_SIZE_4, STRIDE_4)
#define KERNEL_SIZE_5 3
#define STRIDE_5 1
#define PADDING_5 1
#define NUM_FILTERS_5 64

// Layer 6 - Max pooling
#define INPUT_FILTERS_6 NUM_FILTERS_5
#define INPUT_ROWS_6 COMPUTE_OUTPUT_SIZE(INPUT_ROWS_5, PADDING_5, KERNEL_SIZE_5, STRIDE_5)
#define INPUT_COLS_6 COMPUTE_OUTPUT_SIZE(INPUT_COLS_5, PADDING_5, KERNEL_SIZE_5, STRIDE_5)
#define KERNEL_SIZE_6 2
#define STRIDE_6 2
#define PADDING_6 0
#define NUM_FILTERS_6 INPUT_FILTERS_6

// NOTE: We will flatten feature map

// Layer 7 - Fully Connected Layer + relu
#define INPUT_COLS_7 NUM_FILTERS_6 * 4 * 4
#define WEIGHT_ROWS_7 100
#define WEIGHT_COLS_7 NUM_FILTERS_6 * 4 * 4
#define NUM_FILTERS_7 100

// Layer 8 - Fully Connected Layer + relu
#define INPUT_COLS_8 WEIGHT_ROWS_7
#define WEIGHT_ROWS_8 50
#define WEIGHT_COLS_8 NUM_FILTERS_7
#define NUM_FILTERS_8 50

// Layer 9 - Fully Connected Layer
#define INPUT_COLS_9 WEIGHT_ROWS_8
#define WEIGHT_ROWS_9 10
#define WEIGHT_COLS_9 NUM_FILTERS_8
#define NUM_FILTERS_9 10

// Layer 10 - Output layer
#define TOTAL_CLASSES 10

typedef struct Params {
    float weights1[NUM_FILTERS_1][INPUT_FILTERS_1][KERNEL_SIZE_1][KERNEL_SIZE_1];
    float biases1[NUM_FILTERS_1];
    float weights2[NUM_FILTERS_3][INPUT_FILTERS_3][KERNEL_SIZE_3][KERNEL_SIZE_3];
    float biases2[NUM_FILTERS_3];
    float weights3[NUM_FILTERS_5][INPUT_FILTERS_5][KERNEL_SIZE_5][KERNEL_SIZE_5];
    float biases3[NUM_FILTERS_5];
    float weights4[WEIGHT_ROWS_7][WEIGHT_COLS_7];
    float biases4[NUM_FILTERS_7];
    float weights5[WEIGHT_ROWS_8][WEIGHT_COLS_8];
    float biases5[NUM_FILTERS_8];
    float weights6[WEIGHT_ROWS_9][WEIGHT_COLS_9];
    float biases6[NUM_FILTERS_9];
} Params;

typedef struct ImageData {
    // float image[MAX_IMAGE_HEIGHT][MAX_IMAGE_WIDTH][MAX_IMAGE_CHANNELS][MAX_IMAGE_FILTERS];
    float image[INPUT_FILTERS_1][INPUT_ROWS_1 + 2 * PADDING_1][INPUT_COLS_1 + 2 * PADDING_1];
    float layer_1[INPUT_FILTERS_2][INPUT_ROWS_2 + 2 * PADDING_2][INPUT_COLS_2 + 2 * PADDING_2];
    float layer_2[INPUT_FILTERS_3][INPUT_ROWS_3 + 2 * PADDING_3][INPUT_COLS_3 + 2 * PADDING_3];
    float layer_3[INPUT_FILTERS_4][INPUT_ROWS_4 + 2 * PADDING_4][INPUT_COLS_4 + 2 * PADDING_4];
    float layer_4[INPUT_FILTERS_5][INPUT_ROWS_5 + 2 * PADDING_5][INPUT_COLS_5 + 2 * PADDING_5];
    float layer_5[INPUT_FILTERS_6][INPUT_ROWS_6 + 2 * PADDING_6][INPUT_COLS_6 + 2 * PADDING_6];
    float layer_6[INPUT_COLS_7];
    float layer_7[INPUT_COLS_8];
    float layer_8[INPUT_COLS_9];
    float layer_9[TOTAL_CLASSES];
    int filters;
    int width;
    int height;
} ImageData;

int forwardPass(ImageData& inputData, Params& param);

void layer_1_conv(ImageData& inputData, Params& param, int padding, int further_padding, int stride, int kernel_size, int out_filters,
                  int in_filters);

void layer_2_max_pool(ImageData& inputData, int padding, int further_padding, int stride, int kernel_size, int in_filters);

void layer_3_conv(ImageData& inputData, Params& param, int padding, int further_padding, int stride, int kernel_size, int out_filters,
                  int in_filters);

void layer_4_max_pool(ImageData& inputData, int padding, int further_padding, int stride, int kernel_size, int in_filters);

void layer_5_conv(ImageData& inputData, Params& param, int padding, int further_padding, int stride, int kernel_size, int out_filters,
                  int in_filters);

void layer_6_max_pool_flatten(ImageData& inputData, int padding, int stride, int kernel_size, int out_filters, int in_filters);

void layer_7_fc(ImageData& inputData, Params& param, int rows, int cols);

void layer_8_fc(ImageData& inputData, Params& param, int rows, int cols);

void layer_9_fc(ImageData& inputData, Params& param, int rows, int cols);

void loadDataset(const char* folderPath, ImageData& imageData, int& test_set_size, int padding, Params& param, cv::Mat& image, int &correct_cases);

void loadParams(const char* paramPath, Params& params);

#endif // CNN_H
