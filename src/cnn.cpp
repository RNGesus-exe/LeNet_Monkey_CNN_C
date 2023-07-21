#include "../include/cnn.h"

void loadDataset(const char* folderPath, ImageData& imageData, int& test_set_size, int padding, Params& param, cv::Mat& image) {
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
            loadDataset(subfolderPath, imageData, test_set_size, padding, param, image);
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
                imageData.low_height = 0;
                imageData.low_width = 0;

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

                // Send the image data to forward pass
                forwardPass(imageData, param);

                // display the image to test if loaded correctly in 3D matrix
                // cv::Mat cvImage(imageData.height, imageData.width, CV_8UC3);

                // for (int row = 0; row < imageData.height; ++row) {
                //     for (int col = 0; col < imageData.width; ++col) {
                //         cv::Vec3b& pixel = cvImage.at<cv::Vec3b>(row, col);
                //         for (int ch = 0; ch < imageData.channels; ++ch) {
                //             pixel[ch] = imageData.image[row][col][ch];
                //        }
                //     }
                // }
                // Display the image using OpenCV
                // cv::imshow("Image", cvImage);
                // cv::waitKey(0); // Wait for key press

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

    fclose(file);
}

int forwardPass(ImageData& inputData, Params& param) {

    FILE* file = fopen("input_map.txt", "wb");
    for (int f = 0; f < inputData.filters; f++) {
        for (int i = 0; i < inputData.height; i++) {
            for (int j = 0; j < inputData.width; j++) {
                fprintf(file, "%f ", inputData.image[f][i][j]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n\n");
    }
    fclose(file);

    layer_1_conv(inputData, param, PADDING_1, PADDING_2, STRIDE_1, KERNEL_SIZE_1, NUM_FILTERS_1, INPUT_FILTERS_1);

    file = fopen("layer_1.txt", "wb");
    for (int f = 0; f < inputData.filters; f++) {
        for (int i = 0; i < inputData.height; i++) {
            for (int j = 0; j < inputData.width; j++) {
                fprintf(file, "%f ", inputData.layer_1[f][i][j]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n\n");
    }
    fclose(file);

    layer_1_conv(inputData, param, PADDING_1, PADDING_2, STRIDE_1, KERNEL_SIZE_1, NUM_FILTERS_1, INPUT_FILTERS_1);

    file = fopen("weights_bias.txt", "wb");
    for (int f = 0; f < NUM_FILTERS_1; f++) {
        for (int c = 0; c < INPUT_FILTERS_1; c++) {
            for (int i = 0; i < KERNEL_SIZE_1; i++) {
                for (int j = 0; j < KERNEL_SIZE_1; j++) {
                    fprintf(file, "%f ", param.weights1[f][c][i][j]);
                }
                fprintf(file, "\n");
            }
            fprintf(file, "\n\n");
        }
        fprintf(file, "\n\n\n");
    }
    fprintf(file, "\n\n\n\n");

    for (int f = 0; f < NUM_FILTERS_1; f++) {
        fprintf(file, "%f ", param.biases1[f]);
    }
    fclose(file);
    return 0;
}

void layer_2_conv(ImageData& imageData, Params& param, int padding, int new_pading, int stride, int kernel_size, int out_filters,
                  int in_filters) {

    imageData.height = COMPUTE_OUTPUT_SIZE(imageData.height - 2 * padding, padding, kernel_size, stride);
    imageData.width = COMPUTE_OUTPUT_SIZE(imageData.width - 2 * padding, padding, kernel_size, stride);
    imageData.filters = out_filters;
    int stride_x, stride_y;

    for (int out_f = 0; out_f < imageData.filters; out_f++) {
        for (int row = 0; row < imageData.height; row++) {
            for (int col = 0; col < imageData.width; col++) {

                imageData.layer_1[out_f][row][col] = 0;
                for (int in_f = 0; in_f < in_filters; in_f++) {

                    stride_x = stride * row;
                    stride_y = stride * col;

                    for (int i = 0; i < kernel_size; i++) {
                        for (int j = 0; j < kernel_size; j++) {

                            imageData.layer_1[out_f][row][col] +=
                                imageData.image[in_f][stride_x + i][stride_y + j] * param.weights1[out_f][in_f][i][j];
                        }
                    }
                }
                imageData.layer_1[out_f][row][col] = relu(imageData.layer_1[out_f][row][col] + param.biases1[out_f]);
            }
        }
    }
}
void layer_1_conv(ImageData& imageData, Params& param, int padding, int new_pading, int stride, int kernel_size, int out_filters,
                  int in_filters) {

    imageData.height = COMPUTE_OUTPUT_SIZE(imageData.height - 2 * padding, padding, kernel_size, stride);
    imageData.width = COMPUTE_OUTPUT_SIZE(imageData.width - 2 * padding, padding, kernel_size, stride);
    imageData.filters = out_filters;
    int stride_x, stride_y;

    for (int out_f = 0; out_f < imageData.filters; out_f++) {
        for (int row = 0; row < imageData.height; row++) {
            for (int col = 0; col < imageData.width; col++) {

                imageData.layer_1[out_f][row][col] = 0;
                for (int in_f = 0; in_f < in_filters; in_f++) {

                    stride_x = stride * row;
                    stride_y = stride * col;

                    for (int i = 0; i < kernel_size; i++) {
                        for (int j = 0; j < kernel_size; j++) {

                            imageData.layer_1[out_f][row][col] +=
                                imageData.image[in_f][stride_x + i][stride_y + j] * param.weights1[out_f][in_f][i][j];
                        }
                    }
                }
                imageData.layer_1[out_f][row][col] = relu(imageData.layer_1[out_f][row][col] + param.biases1[out_f]);
            }
        }
    }
}
