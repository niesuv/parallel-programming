#include "data_loader.h"
#include <svm.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <chrono>
#include <vector>

float *loadFeatures(const std::string &filename, int num_images)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open feature file: " << filename << std::endl;
        return nullptr;
    }

    float *features = new float[num_images * 8192];
    file.read(reinterpret_cast<char *>(features), num_images * 8192 * sizeof(float));
    file.close();

    std::cout << "Loaded " << num_images << " feature vectors ("
              << num_images * 8192 * sizeof(float) / (1024.0f * 1024.0f) << " MB)" << std::endl;

    return features;
}

void trainAndEvaluateSVM(const std::string &train_features_file,
                         const std::string &test_features_file,
                         CIFAR10Dataset *dataset)
{
    std::cout << "\n=== SVM Training and Evaluation ===" << std::endl;

    // Load features
    std::cout << "Loading training features..." << std::endl;
    float *train_features = loadFeatures(train_features_file, 50000);
    if (!train_features)
        return;

    std::cout << "Loading test features..." << std::endl;
    float *test_features = loadFeatures(test_features_file, 10000);
    if (!test_features)
    {
        delete[] train_features;
        return;
    }

    // Prepare LIBSVM problem for training
    std::cout << "\nPreparing SVM training data..." << std::endl;
    svm_problem train_prob;
    train_prob.l = 50000;
    train_prob.y = new double[50000];
    train_prob.x = new svm_node *[50000];

    for (int i = 0; i < 50000; i++)
    {
        train_prob.y[i] = dataset->getTrainLabel(i);

        // Create sparse feature vector
        train_prob.x[i] = new svm_node[8193]; // 8192 features + terminator
        for (int j = 0; j < 8192; j++)
        {
            train_prob.x[i][j].index = j + 1;
            train_prob.x[i][j].value = train_features[i * 8192 + j];
        }
        train_prob.x[i][8192].index = -1; // Terminator
    }

    // Set SVM parameters
    std::cout << "Configuring SVM parameters..." << std::endl;
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 1.0f / 8192.0f; // 1 / num_features
    param.coef0 = 0;
    param.C = 10.0;
    param.nu = 0.5;
    param.p = 0.1;
    param.cache_size = 2000; // MB
    param.eps = 0.001;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;

    std::cout << "SVM Parameters:" << std::endl;
    std::cout << "  Kernel: RBF" << std::endl;
    std::cout << "  C: " << param.C << std::endl;
    std::cout << "  Gamma: " << param.gamma << std::endl;
    std::cout << "  Training samples: " << train_prob.l << std::endl;

    // Train SVM
    std::cout << "\nTraining SVM..." << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();
    svm_model *model = svm_train(&train_prob, &param);
    auto train_end = std::chrono::high_resolution_clock::now();

    auto train_time = std::chrono::duration_cast<std::chrono::seconds>(
                          train_end - train_start)
                          .count();

    std::cout << "SVM training completed in " << train_time << "s" << std::endl;

    // Prepare test problem
    std::cout << "\nPreparing test data..." << std::endl;
    svm_problem test_prob;
    test_prob.l = 10000;
    test_prob.y = new double[10000];
    test_prob.x = new svm_node *[10000];

    for (int i = 0; i < 10000; i++)
    {
        test_prob.y[i] = dataset->getTestLabel(i);

        test_prob.x[i] = new svm_node[8193];
        for (int j = 0; j < 8192; j++)
        {
            test_prob.x[i][j].index = j + 1;
            test_prob.x[i][j].value = test_features[i * 8192 + j];
        }
        test_prob.x[i][8192].index = -1;
    }

    // Evaluate on test set
    std::cout << "\nEvaluating on test set..." << std::endl;
    int correct = 0;
    int confusion_matrix[10][10];
    std::memset(confusion_matrix, 0, sizeof(confusion_matrix));

    for (int i = 0; i < 10000; i++)
    {
        double predicted = svm_predict(model, test_prob.x[i]);
        int true_label = (int)test_prob.y[i];

        if (predicted == true_label)
            correct++;
        confusion_matrix[true_label][(int)predicted]++;

        if ((i + 1) % 1000 == 0)
        {
            std::cout << "  Tested " << i + 1 << " samples..." << std::endl;
        }
    }

    float accuracy = 100.0f * correct / 10000;

    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "Correct Predictions: " << correct << " / 10000" << std::endl;

    // Print confusion matrix
    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << "       ";
    for (int i = 0; i < 10; i++)
        std::cout << std::setw(6) << i;
    std::cout << std::endl;

    for (int i = 0; i < 10; i++)
    {
        std::cout << "True " << std::setw(1) << i << ":";
        for (int j = 0; j < 10; j++)
        {
            std::cout << std::setw(6) << confusion_matrix[i][j];
        }
        std::cout << std::endl;
    }

    // Per-class accuracy
    std::cout << "\nPer-class Accuracy:" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        int class_total = 0;
        for (int j = 0; j < 10; j++)
        {
            class_total += confusion_matrix[i][j];
        }
        if (class_total > 0)
        {
            float class_acc = 100.0f * confusion_matrix[i][i] / class_total;
            std::cout << "Class " << i << " (" << std::setw(10) << dataset->getClassName(i) << "): "
                      << std::fixed << std::setprecision(2) << class_acc << "%" << std::endl;
        }
    }

    // Cleanup
    std::cout << "\nCleaning up..." << std::endl;
    svm_free_and_destroy_model(&model);

    for (int i = 0; i < 50000; i++)
        delete[] train_prob.x[i];
    delete[] train_prob.x;
    delete[] train_prob.y;

    for (int i = 0; i < 10000; i++)
        delete[] test_prob.x[i];
    delete[] test_prob.x;
    delete[] test_prob.y;

    delete[] train_features;
    delete[] test_features;

    std::cout << "Done!\n"
              << std::endl;
}

int main(int argc, char **argv)
{
    // Default values
    std::string data_dir = "./data/cifar-10-batches-bin";
    std::string train_features_file = "../build/cifar10_features.bin";
    std::string test_features_file = "../build/cifar10_features.bin";

    // Parse command-line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc)
        {
            data_dir = argv[++i];
        }
        else if (strcmp(argv[i], "--train-features") == 0 && i + 1 < argc)
        {
            train_features_file = argv[++i];
        }
        else if (strcmp(argv[i], "--test-features") == 0 && i + 1 < argc)
        {
            test_features_file = argv[++i];
        }
        else if (strcmp(argv[i], "--features") == 0 && i + 1 < argc)
        {
            train_features_file = argv[++i];
            test_features_file = argv[i]; // Use same file for both
        }
        else if (strcmp(argv[i], "--help") == 0)
        {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "  --data-dir PATH        Path to CIFAR-10 data directory" << std::endl;
            std::cout << "  --features FILE        Features file containing all extracted features" << std::endl;
            std::cout << "  --train-features FILE  Training features file" << std::endl;
            std::cout << "  --test-features FILE   Test features file" << std::endl;
            std::cout << "  --help                 Show this help message" << std::endl;
            return 0;
        }
    }

    // Load dataset for labels
    CIFAR10Dataset dataset;

    if (!dataset.loadTrainingData(data_dir))
    {
        std::cerr << "Failed to load training data from " << data_dir << std::endl;
        return 1;
    }

    if (!dataset.loadTestData(data_dir))
    {
        std::cerr << "Failed to load test data from " << data_dir << std::endl;
        return 1;
    }

    if (!dataset.loadClassNames(data_dir + "/batches.meta.txt"))
    {
        std::cerr << "Failed to load class names" << std::endl;
        return 1;
    }

    // Train and evaluate SVM
    trainAndEvaluateSVM(train_features_file, test_features_file, &dataset);

    return 0;
}
