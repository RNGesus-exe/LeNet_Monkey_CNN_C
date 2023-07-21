#include "../include/cnn.h"

int main() {
    int test_set_size = 0;
    Params param;
    ImageData inputImage;
    ImageData ouputImage;
    cv::Mat image;

    const char* images_path = "../extern/single_test_data";
    const char* param_path = "../extern/parameters.txt";

    loadParams(param_path, param);
    loadDataset(images_path, inputImage, test_set_size, PADDING_1, param, image);

    printf("Images = %d\n", test_set_size);

    return 0;
}
