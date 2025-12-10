#include <iostream>
#include <iomanip>
#include "../include/data_loader.h"

int main(int argc, char **argv)
{
    std::string data_dir = "./data/cifar-10-batches-bin";
    std::string meta = data_dir + "/batches.meta.txt";

    if (argc > 1)
        data_dir = argv[1];
    if (argc > 2)
        meta = argv[2];

    CIFAR10Dataset ds;
    std::cout << "Loading class names from: " << meta << std::endl;
    if (!ds.loadClassNames(meta))
    {
        std::cerr << "Failed to load class names (continue if file missing)." << std::endl;
    }

    std::cout << "Loading training data from: " << data_dir << std::endl;
    if (!ds.loadTrainingData(data_dir))
    {
        std::cerr << "Training data load failed. Ensure CIFAR-10 binary files are present." << std::endl;
        return 1;
    }

    std::cout << "Loading test data from: " << data_dir << std::endl;
    if (!ds.loadTestData(data_dir))
    {
        std::cerr << "Test data load failed." << std::endl;
        return 1;
    }

    std::cout << "Train size: " << ds.getTrainSize() << std::endl;
    std::cout << "Test size: " << ds.getTestSize() << std::endl;

    // Get first batch
    int batch_size = 4;
    int image_size = 3 * 32 * 32;
    float *batch_images = new float[batch_size * image_size];
    uint8_t *batch_labels = new uint8_t[batch_size];

    ds.getBatch(0, batch_size, batch_images, batch_labels);

    std::cout << "First batch labels: ";
    for (int i = 0; i < batch_size; ++i)
        std::cout << (int)batch_labels[i] << " ";
    std::cout << std::endl;

    // Print first image's first 10 pixel values (channel 0)
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "First image (first 10 pixels channel 0): ";
    for (int p = 0; p < 10; ++p)
        std::cout << batch_images[p] << " ";
    std::cout << std::endl;

    delete[] batch_images;
    delete[] batch_labels;

    std::cout << "Data loader test completed." << std::endl;
    return 0;
}
