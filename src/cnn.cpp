#include "../include/cnn.h"

const char* monkey_classes[] = {"Emperor Tamarin", "Gray Langur", "Hamadryas Baboon", "Proboscis Monkey", "Vervet Monkey",
                                "Golden Monkey",   "Mandril",     "Bald Uakari",      "White Faced Saki", "Red Howler"};

void loadDataset(const char* folderPath, ImageData& imageData, int& test_set_size, int padding, Params& param, cv::Mat& image,
                 int& correct_cases) {
    DIR* directory;
    struct dirent* entry;

    // Open the directory
    directory = opendir(folderPath);
    if (directory == NULL) {
        fprintf(stderr, "Failed to open directory: %s\n", folderPath);
        return;
    }

    // Read the directory entries
    while ((entry = readdir(directory)) != NULL) {
        if (entry->d_type == DT_DIR) { // Check if it's a subdirectory
            // Ignore the current directory (.) and parent directory (..)
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }

            // Construct the subfolder path
            char subfolderPath[MAX_PATH_LENGTH];
            strncpy(subfolderPath, folderPath, sizeof(subfolderPath));
            strncat(subfolderPath, "/", sizeof(subfolderPath) - strlen(subfolderPath) - 1);
            strncat(subfolderPath, entry->d_name, sizeof(subfolderPath) - strlen(subfolderPath) - 1);

            // Read images recursively in the subfolder
            loadDataset(subfolderPath, imageData, test_set_size, padding, param, image, correct_cases);
        } else if (entry->d_type == DT_REG) { // Check if it's a regular file
            // Get the file name
            const char* fileName = entry->d_name;

            // Check if the file is an image (you can add more image formats if needed)
            const char* extension = strrchr(fileName, '.');
            if (extension != NULL && (strcasecmp(extension, ".jpg") == 0 || strcasecmp(extension, ".jpeg") == 0)) {

                // Construct the full image path
                char imagePath[MAX_PATH_LENGTH];
                snprintf(imagePath, sizeof(imagePath), "%s/%s", folderPath, fileName);
                // std::cout << imagePath << std::endl;

                // Read the image
                image = cv::imread(imagePath, cv::IMREAD_COLOR);

                if (image.empty()) {
                    std::cout << "Error: Could not read the image.\n";
                    return;
                }

                // Resize the image to 124x124
                cv::Size newSize(INPUT_ROWS_1, INPUT_COLS_1);
                cv::resize(image, image, newSize);
                // image = bilinearInterpolation(image, 128, 128);

                // Convert the resized image to a 3x124x124 array
                cv::Mat channels[INPUT_FILTERS_1];
                cv::split(image, channels);

                for (int f = 0; f < INPUT_FILTERS_1; f++) {
                    for (int i = 0; i < INPUT_ROWS_1; i++) {
                        for (int j = 0; j < INPUT_COLS_1; j++) {
                            imageData.image[INPUT_FILTERS_1 - f - 1][padding + i][padding + j] =
                                (static_cast<float>(channels[f].at<uchar>(i, j)) - MEAN) / STD;
                        }
                    }
                }

                imageData.height = INPUT_ROWS_1 + 2 * padding;
                imageData.width = INPUT_COLS_1 + 2 * padding;
                imageData.filters = INPUT_FILTERS_1;

                // Apply Padding
                for (int f = 0; f < imageData.filters; f++) {
                    for (int i = 0; i < imageData.height; i++) {
                        for (int j = 0; j < padding; j++) {
                            imageData.image[f][i][j] = 0;
                            imageData.image[f][imageData.height - i - 1][j] = 0;
                        }
                    }
                }
                for (int f = 0; f < imageData.filters; f++) {
                    for (int i = 0; i < padding; i++) {
                        for (int j = 0; j < imageData.width; j++) {
                            imageData.image[f][i][j] = 0;
                            imageData.image[f][i][imageData.width - j - 1] = 0;
                        }
                    }
                }

                const char* class_name = strrchr(folderPath, '/') + 1;
                // Send the image data to forward pass
                int res = forwardPass(imageData, param);

                if (strcmp(class_name, monkey_classes[res]) == 0) {
                    correct_cases++;
                }

                // Count total images processed so far
                test_set_size++;
                // printf("Image Path: %s\n", imagePath);
            }
        }
    }

    // Close the directory
    closedir(directory);
}

void loadParams(const char* paramPath, Params& params) {
    FILE* file = fopen(paramPath, "rb");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return;
    }
    char line[MAX_LINE_SIZE];

    // Read weights of layer 1
    if (fgets(line, MAX_LINE_SIZE, file)) {

        char* token;
        token = strtok(line, " ");

        for (int b = 0; b < NUM_FILTERS_1; b++) {
            for (int c = 0; c < INPUT_FILTERS_1; c++) {
                for (int i = 0; i < KERNEL_SIZE_1; i++) {
                    for (int j = 0; j < KERNEL_SIZE_1; j++) {
                        params.weights1[b][c][i][j] = atof(token);
                        token = strtok(NULL, " ");
                    }
                }
            }
        }
    }

    // Read biases of layer 1
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_1; c++) {
            params.biases1[c] = atof(token);
            token = strtok(NULL, " ");
        }
    }

    // Read weights of layer 3
    if (fgets(line, MAX_LINE_SIZE, file)) {

        char* token;
        token = strtok(line, " ");

        for (int b = 0; b < NUM_FILTERS_3; b++) {
            for (int c = 0; c < INPUT_FILTERS_3; c++) {
                for (int i = 0; i < KERNEL_SIZE_3; i++) {
                    for (int j = 0; j < KERNEL_SIZE_3; j++) {
                        params.weights2[b][c][i][j] = atof(token);
                        token = strtok(NULL, " ");
                    }
                }
            }
        }
    }

    // Read biases of layer 3
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_3; c++) {
            params.biases2[c] = atof(token);
            token = strtok(NULL, " ");
        }
    }

    // Read weights of layer 5
    if (fgets(line, MAX_LINE_SIZE, file)) {

        char* token;
        token = strtok(line, " ");

        for (int b = 0; b < NUM_FILTERS_5; b++) {
            for (int c = 0; c < INPUT_FILTERS_5; c++) {
                for (int i = 0; i < KERNEL_SIZE_5; i++) {
                    for (int j = 0; j < KERNEL_SIZE_5; j++) {
                        params.weights3[b][c][i][j] = atof(token);
                        token = strtok(NULL, " ");
                    }
                }
            }
        }
    }

    // Read biases of layer 5
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < NUM_FILTERS_5; c++) {
            params.biases3[c] = atof(token);
            token = strtok(NULL, " ");
        }
    }

    // Read weights of layer 6
    if (fgets(line, MAX_LINE_SIZE, file)) {

        char* token;
        token = strtok(line, " ");

        for (int c = 0; c < WEIGHT_ROWS_7; c++) {
            for (int i = 0; i < WEIGHT_COLS_7; i++) {
                params.weights4[c][i] = atof(token);
                token = strtok(NULL, " ");
            }
        }
    }

    // Read biases of layer 6
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < WEIGHT_ROWS_7; c++) {
            params.biases4[c] = atof(token);
            token = strtok(NULL, " ");
        }
    }

    // Read weights of layer 7
    if (fgets(line, MAX_LINE_SIZE, file)) {

        char* token;
        token = strtok(line, " ");

        for (int c = 0; c < WEIGHT_ROWS_8; c++) {
            for (int i = 0; i < WEIGHT_COLS_8; i++) {
                params.weights5[c][i] = atof(token);
                token = strtok(NULL, " ");
            }
        }
    }

    // Read biases of layer 7
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < WEIGHT_ROWS_8; c++) {
            params.biases5[c] = atof(token);
            token = strtok(NULL, " ");
        }
    }

    // Read weights of layer 8
    if (fgets(line, MAX_LINE_SIZE, file)) {

        char* token;
        token = strtok(line, " ");

        for (int c = 0; c < WEIGHT_ROWS_9; c++) {
            for (int i = 0; i < WEIGHT_COLS_9; i++) {
                params.weights6[c][i] = atof(token);
                token = strtok(NULL, " ");
            }
        }
    }

    // Read biases of layer 8
    if (fgets(line, MAX_LINE_SIZE, file)) {
        char* token;
        token = strtok(line, " ");
        for (int c = 0; c < WEIGHT_ROWS_9; c++) {
            params.biases6[c] = atof(token);
            token = strtok(NULL, " ");
        }
    }

    fclose(file);
}

int forwardPass(ImageData& inputData, Params& param) {

    layer_1_conv(inputData, param, PADDING_1, PADDING_2, STRIDE_1, KERNEL_SIZE_1, NUM_FILTERS_1, INPUT_FILTERS_1);
    layer_2_max_pool(inputData, PADDING_2, PADDING_3, STRIDE_2, KERNEL_SIZE_2, NUM_FILTERS_2);
    layer_3_conv(inputData, param, PADDING_3, PADDING_4, STRIDE_3, KERNEL_SIZE_3, NUM_FILTERS_3, INPUT_FILTERS_3);
    layer_4_max_pool(inputData, PADDING_4, PADDING_5, STRIDE_4, KERNEL_SIZE_4, NUM_FILTERS_4);
    layer_5_conv(inputData, param, PADDING_5, PADDING_6, STRIDE_5, KERNEL_SIZE_5, NUM_FILTERS_5, INPUT_FILTERS_5);
    layer_6_max_pool_flatten(inputData, PADDING_6, STRIDE_6, KERNEL_SIZE_6, INPUT_COLS_7, NUM_FILTERS_6);
    layer_7_fc(inputData, param, WEIGHT_ROWS_7, WEIGHT_COLS_7);
    layer_8_fc(inputData, param, WEIGHT_ROWS_8, WEIGHT_COLS_8);
    layer_9_fc(inputData, param, WEIGHT_ROWS_9, WEIGHT_COLS_9);

    int max_ind = 0;
    for (int i = 1; i < TOTAL_CLASSES; i++) {
        if (inputData.layer_9[i] > inputData.layer_9[max_ind]) {
            max_ind = i;
        }
    }

    return max_ind;
}

void layer_9_fc(ImageData& imageData, Params& param, int rows, int cols) {
    for (int n = 0; n < rows; n++) {
        imageData.layer_9[n] = 0;

        for (int i = 0; i < cols; i++) {
            imageData.layer_9[n] += imageData.layer_8[i] * param.weights6[n][i];
        }

        imageData.layer_9[n] = imageData.layer_9[n] + param.biases6[n];
    }
}

void layer_8_fc(ImageData& imageData, Params& param, int rows, int cols) {
    for (int n = 0; n < rows; n++) {
        imageData.layer_8[n] = 0;

        for (int i = 0; i < cols; i++) {
            imageData.layer_8[n] += imageData.layer_7[i] * param.weights5[n][i];
        }

        imageData.layer_8[n] = relu(imageData.layer_8[n] + param.biases5[n]);
    }
}

void layer_7_fc(ImageData& imageData, Params& param, int rows, int cols) {
    for (int n = 0; n < rows; n++) {
        imageData.layer_7[n] = 0;

        for (int i = 0; i < cols; i++) {
            imageData.layer_7[n] += imageData.layer_6[i] * param.weights4[n][i];
        }

        imageData.layer_7[n] = relu(imageData.layer_7[n] + param.biases4[n]);
    }
}

void layer_6_max_pool_flatten(ImageData& imageData, int padding, int stride, int kernel_size, int out_filters, int in_filters) {

    imageData.height = COMPUTE_OUTPUT_SIZE(imageData.height - 2 * padding, padding, kernel_size, stride);
    imageData.width = COMPUTE_OUTPUT_SIZE(imageData.width - 2 * padding, padding, kernel_size, stride);
    imageData.filters = out_filters;
    int stride_x, stride_y;

    int ind = 0;
    for (int in_f = 0; in_f < in_filters; in_f++) {
        for (int row = 0; row < imageData.height; row++) {
            for (int col = 0; col < imageData.width; col++) {

                stride_x = stride * row;
                stride_y = stride * col;
                float max_val = INT32_MIN;

                for (int i = 0; i < kernel_size; i++) {
                    for (int j = 0; j < kernel_size; j++) {
                        if (imageData.layer_5[in_f][stride_x + i][stride_y + j] > max_val) {
                            max_val = imageData.layer_5[in_f][stride_x + i][stride_y + j];
                        }
                    }
                }
                imageData.layer_6[ind++] = max_val;
            }
        }
    }
}

void layer_5_conv(ImageData& imageData, Params& param, int padding, int new_padding, int stride, int kernel_size, int out_filters,
                  int in_filters) {

    imageData.height = COMPUTE_OUTPUT_SIZE(imageData.height - 2 * padding, padding, kernel_size, stride);
    imageData.width = COMPUTE_OUTPUT_SIZE(imageData.width - 2 * padding, padding, kernel_size, stride);
    imageData.filters = out_filters;
    int stride_x, stride_y;

    for (int j = 0; j < new_padding; j++) {
        for (int f = 0; f < imageData.filters; f++) {
            for (int i = 0; i < imageData.height; i++) {
                imageData.layer_5[f][i][j] = 0;
                imageData.layer_5[f][imageData.height - i - 1][j] = 0;
            }
        }
    }

    for (int i = 0; i < new_padding; i++) {
        for (int j = 0; j < imageData.width; j++) {
            for (int f = 0; f < imageData.filters; f++) {
                imageData.layer_5[f][i][j] = 0;
                imageData.layer_5[f][i][imageData.width - j - 1] = 0;
            }
        }
    }

    for (int out_f = 0; out_f < imageData.filters; out_f++) {
        for (int row = 0; row < imageData.height; row++) {
            for (int col = 0; col < imageData.width; col++) {

                imageData.layer_5[out_f][row][col] = 0;
                for (int in_f = 0; in_f < in_filters; in_f++) {

                    stride_x = stride * row;
                    stride_y = stride * col;

                    for (int i = 0; i < kernel_size; i++) {
                        for (int j = 0; j < kernel_size; j++) {

                            imageData.layer_5[out_f][row + new_padding][col + new_padding] +=
                                imageData.layer_4[in_f][stride_x + i][stride_y + j] * param.weights3[out_f][in_f][i][j];
                        }
                    }
                }
                imageData.layer_5[out_f][row + new_padding][col + new_padding] =
                    relu(imageData.layer_5[out_f][row][col] + param.biases3[out_f]);
            }
        }
    }

    imageData.height += 2 * new_padding;
    imageData.width += 2 * new_padding;
}

void layer_4_max_pool(ImageData& imageData, int padding, int new_padding, int stride, int kernel_size, int out_filters) {

    imageData.height = COMPUTE_OUTPUT_SIZE(imageData.height - 2 * padding, padding, kernel_size, stride);
    imageData.width = COMPUTE_OUTPUT_SIZE(imageData.width - 2 * padding, padding, kernel_size, stride);
    imageData.filters = out_filters;
    int stride_x, stride_y;

    for (int j = 0; j < new_padding; j++) {
        for (int f = 0; f < imageData.filters; f++) {
            for (int i = 0; i < imageData.height; i++) {
                imageData.layer_4[f][i][j] = 0;
                imageData.layer_4[f][imageData.height - i - 1][j] = 0;
            }
        }
    }

    for (int i = 0; i < new_padding; i++) {
        for (int j = 0; j < imageData.width; j++) {
            for (int f = 0; f < imageData.filters; f++) {
                imageData.layer_4[f][i][j] = 0;
                imageData.layer_4[f][i][imageData.width - j - 1] = 0;
            }
        }
    }

    for (int out_f = 0; out_f < imageData.filters; out_f++) {
        for (int row = 0; row < imageData.height; row++) {
            for (int col = 0; col < imageData.width; col++) {

                stride_x = stride * row;
                stride_y = stride * col;
                float max_val = INT32_MIN;

                for (int i = 0; i < kernel_size; i++) {
                    for (int j = 0; j < kernel_size; j++) {
                        if (imageData.layer_3[out_f][stride_x + i][stride_y + j] > max_val) {
                            max_val = imageData.layer_3[out_f][stride_x + i][stride_y + j];
                        }
                    }
                }
                imageData.layer_4[out_f][row + new_padding][col + new_padding] = max_val;
            }
        }
    }

    imageData.height += 2 * new_padding;
    imageData.width += 2 * new_padding;
}

void layer_3_conv(ImageData& imageData, Params& param, int padding, int new_padding, int stride, int kernel_size, int out_filters,
                  int in_filters) {

    imageData.height = COMPUTE_OUTPUT_SIZE(imageData.height - 2 * padding, padding, kernel_size, stride);
    imageData.width = COMPUTE_OUTPUT_SIZE(imageData.width - 2 * padding, padding, kernel_size, stride);
    imageData.filters = out_filters;
    int stride_x, stride_y;

    for (int j = 0; j < new_padding; j++) {
        for (int f = 0; f < imageData.filters; f++) {
            for (int i = 0; i < imageData.height; i++) {
                imageData.layer_3[f][i][j] = 0;
                imageData.layer_3[f][imageData.height - i - 1][j] = 0;
            }
        }
    }

    for (int i = 0; i < new_padding; i++) {
        for (int j = 0; j < imageData.width; j++) {
            for (int f = 0; f < imageData.filters; f++) {
                imageData.layer_3[f][i][j] = 0;
                imageData.layer_3[f][i][imageData.width - j - 1] = 0;
            }
        }
    }

    for (int out_f = 0; out_f < imageData.filters; out_f++) {
        for (int row = 0; row < imageData.height; row++) {
            for (int col = 0; col < imageData.width; col++) {

                imageData.layer_3[out_f][row][col] = 0;
                for (int in_f = 0; in_f < in_filters; in_f++) {

                    stride_x = stride * row;
                    stride_y = stride * col;

                    for (int i = 0; i < kernel_size; i++) {
                        for (int j = 0; j < kernel_size; j++) {

                            imageData.layer_3[out_f][row + new_padding][col + new_padding] +=
                                imageData.layer_2[in_f][stride_x + i][stride_y + j] * param.weights2[out_f][in_f][i][j];
                        }
                    }
                }
                imageData.layer_3[out_f][row + new_padding][col + new_padding] =
                    relu(imageData.layer_3[out_f][row][col] + param.biases2[out_f]);
            }
        }
    }

    imageData.height += 2 * new_padding;
    imageData.width += 2 * new_padding;
}

void layer_2_max_pool(ImageData& imageData, int padding, int new_padding, int stride, int kernel_size, int out_filters) {

    imageData.height = COMPUTE_OUTPUT_SIZE(imageData.height - 2 * padding, padding, kernel_size, stride);
    imageData.width = COMPUTE_OUTPUT_SIZE(imageData.width - 2 * padding, padding, kernel_size, stride);
    imageData.filters = out_filters;
    int stride_x, stride_y;

    for (int j = 0; j < new_padding; j++) {
        for (int f = 0; f < imageData.filters; f++) {
            for (int i = 0; i < imageData.height; i++) {
                imageData.layer_2[f][i][j] = 0;
                imageData.layer_2[f][imageData.height - i - 1][j] = 0;
            }
        }
    }

    for (int i = 0; i < new_padding; i++) {
        for (int j = 0; j < imageData.width; j++) {
            for (int f = 0; f < imageData.filters; f++) {
                imageData.layer_2[f][i][j] = 0;
                imageData.layer_2[f][i][imageData.width - j - 1] = 0;
            }
        }
    }

    for (int out_f = 0; out_f < imageData.filters; out_f++) {
        for (int row = 0; row < imageData.height; row++) {
            for (int col = 0; col < imageData.width; col++) {

                stride_x = stride * row;
                stride_y = stride * col;
                float max_val = INT32_MIN;

                for (int i = 0; i < kernel_size; i++) {
                    for (int j = 0; j < kernel_size; j++) {
                        if (imageData.layer_1[out_f][stride_x + i][stride_y + j] > max_val) {
                            max_val = imageData.layer_1[out_f][stride_x + i][stride_y + j];
                        }
                    }
                }
                imageData.layer_2[out_f][row + new_padding][col + new_padding] = max_val;
            }
        }
    }

    imageData.height += 2 * new_padding;
    imageData.width += 2 * new_padding;
}

void layer_1_conv(ImageData& imageData, Params& param, int padding, int new_padding, int stride, int kernel_size, int out_filters,
                  int in_filters) {

    imageData.height = COMPUTE_OUTPUT_SIZE(imageData.height - 2 * padding, padding, kernel_size, stride);
    imageData.width = COMPUTE_OUTPUT_SIZE(imageData.width - 2 * padding, padding, kernel_size, stride);
    imageData.filters = out_filters;
    int stride_x, stride_y;

    for (int j = 0; j < new_padding; j++) {
        for (int f = 0; f < imageData.filters; f++) {
            for (int i = 0; i < imageData.height; i++) {
                imageData.layer_1[f][i][j] = 0;
                imageData.layer_1[f][imageData.height - i - 1][j] = 0;
            }
        }
    }

    for (int i = 0; i < new_padding; i++) {
        for (int j = 0; j < imageData.width; j++) {
            for (int f = 0; f < imageData.filters; f++) {
                imageData.layer_1[f][i][j] = 0;
                imageData.layer_1[f][i][imageData.width - j - 1] = 0;
            }
        }
    }

    for (int out_f = 0; out_f < imageData.filters; out_f++) {
        for (int row = 0; row < imageData.height; row++) {
            for (int col = 0; col < imageData.width; col++) {

                imageData.layer_1[out_f][row][col] = 0;
                for (int in_f = 0; in_f < in_filters; in_f++) {

                    stride_x = stride * row;
                    stride_y = stride * col;

                    for (int i = 0; i < kernel_size; i++) {
                        for (int j = 0; j < kernel_size; j++) {

                            imageData.layer_1[out_f][row + new_padding][col + new_padding] +=
                                imageData.image[in_f][stride_x + i][stride_y + j] * param.weights1[out_f][in_f][i][j];
                        }
                    }
                }
                imageData.layer_1[out_f][row + new_padding][col + new_padding] =
                    relu(imageData.layer_1[out_f][row][col] + param.biases1[out_f]);
            }
        }
    }

    imageData.height += 2 * new_padding;
    imageData.width += 2 * new_padding;
}
