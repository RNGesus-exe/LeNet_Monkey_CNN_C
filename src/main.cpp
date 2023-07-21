#include "../include/cnn.h"

int main() {

    int test_set_size = 0;
    Params param;
    ImageData inputImage;
    ImageData ouputImage;
    cv::Mat image;
    int true_positives = 0;

    const char* images_path = "../extern/test_data";
    const char* param_path = "../extern/parameters.txt";

    loadParams(param_path, param);
    loadDataset(images_path, inputImage, test_set_size, PADDING_1, param, image, true_positives);

    printf("Total Images = %d\n", test_set_size);
    printf("Accuracy = %f\n", (float)true_positives / test_set_size * 100);

    return 0;
}
