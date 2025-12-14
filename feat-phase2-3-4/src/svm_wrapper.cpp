/**
 * @file svm_wrapper.cpp
 * @brief SVM Classifier Wrapper Implementation
 *
 * Provides unified SVM interface with multiple backends:
 *   - WITH_THUNDERSVM: GPU-accelerated SVM (recommended)
 *   - WITH_LIBSVM: CPU-based SVM (fallback)
 *   - Neither: Stub implementation (10% random accuracy)
 *
 * Features:
 *   - Feature normalization (standardization: mean=0, std=1)
 *   - Linear kernel for high-dimensional data
 *   - Automatic gamma calculation for RBF kernel
 */

#include "svm_wrapper.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// =============================================================================
// ThunderSVM Implementation (GPU-accelerated)
// =============================================================================
#ifdef WITH_THUNDERSVM

#include <thundersvm/dataset.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/svmparam.h>

struct SVMWrapper::Impl {
  std::unique_ptr<SVC> model;
  double C = 10.0;
  double gamma = -1.0; // -1 = auto (1/num_features)
  int kernel_type = 2; // RBF

  // Normalization parameters
  std::vector<double> mean;
  std::vector<double> std_dev;
  int feature_dim = 0;

  Impl() = default;
  ~Impl() = default;
};

SVMWrapper::SVMWrapper() : impl_(std::make_unique<Impl>()) {}

SVMWrapper::~SVMWrapper() = default;

void SVMWrapper::set_kernel(int kernel_type) {
  impl_->kernel_type = kernel_type;
}

void SVMWrapper::set_C(double C) { impl_->C = C; }

void SVMWrapper::set_gamma(double gamma) { impl_->gamma = gamma; }

void SVMWrapper::train(const float *features, const int *labels,
                       int num_samples, int feature_dim) {
  std::cout << "Training ThunderSVM (GPU) on " << num_samples << " samples, "
            << feature_dim << " features..." << std::endl;
  std::cout << "  C = " << impl_->C << ", gamma = "
            << (impl_->gamma < 0 ? "auto" : std::to_string(impl_->gamma))
            << std::endl;

  // Normalize features (standardization: mean=0, std=1)
  std::cout << "  Normalizing features..." << std::endl;
  std::vector<float> normalized_features(static_cast<size_t>(num_samples) *
                                         feature_dim);

  // Calculate mean and std for each feature
  std::vector<double> mean(feature_dim, 0.0);
  std::vector<double> std_dev(feature_dim, 0.0);

  for (int i = 0; i < num_samples; ++i) {
    for (int j = 0; j < feature_dim; ++j) {
      mean[j] += features[static_cast<size_t>(i) * feature_dim + j];
    }
  }
  for (int j = 0; j < feature_dim; ++j) {
    mean[j] /= num_samples;
  }

  for (int i = 0; i < num_samples; ++i) {
    for (int j = 0; j < feature_dim; ++j) {
      double diff =
          features[static_cast<size_t>(i) * feature_dim + j] - mean[j];
      std_dev[j] += diff * diff;
    }
  }
  for (int j = 0; j < feature_dim; ++j) {
    std_dev[j] = std::sqrt(std_dev[j] / num_samples);
    if (std_dev[j] < 1e-8)
      std_dev[j] = 1.0; // Avoid division by zero
  }

  // Apply normalization
  for (int i = 0; i < num_samples; ++i) {
    for (int j = 0; j < feature_dim; ++j) {
      normalized_features[static_cast<size_t>(i) * feature_dim + j] =
          static_cast<float>(
              (features[static_cast<size_t>(i) * feature_dim + j] - mean[j]) /
              std_dev[j]);
    }
  }

  // Store normalization params for prediction
  impl_->mean = mean;
  impl_->std_dev = std_dev;
  impl_->feature_dim = feature_dim;

  // Convert labels to float for ThunderSVM
  std::vector<float> labels_float(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    labels_float[i] = static_cast<float>(labels[i]);
  }

  // Create dataset using dense format
  DataSet dataset;
  dataset.load_from_dense(num_samples, feature_dim, normalized_features.data(),
                          labels_float.data());

  // Set up SVM parameters
  SvmParam param;
  param.svm_type = SvmParam::C_SVC;
  param.kernel_type = SvmParam::LINEAR; // LINEAR kernel is better for high-dim
  param.C = static_cast<float>(impl_->C);

  // Auto gamma: 1/num_features
  if (impl_->gamma < 0) {
    param.gamma = 1.0f / feature_dim;
  } else {
    param.gamma = static_cast<float>(impl_->gamma);
  }

  // Memory limit (8GB)
  param.max_mem_size = 8192ULL << 20;

  // Create and train model
  impl_->model = std::make_unique<SVC>();
  impl_->model->train(dataset, param);

  std::cout << "ThunderSVM training complete. Support vectors: "
            << impl_->model->total_sv() << std::endl;
}

std::vector<int> SVMWrapper::predict(const float *features, int num_samples,
                                     int feature_dim) const {
  std::vector<int> predictions(num_samples, -1);

  if (!impl_->model) {
    std::cerr << "Error: SVM model not trained!" << std::endl;
    return predictions;
  }

  // Normalize features using stored mean/std from training
  std::vector<float> normalized_features;
  const float *features_to_use = features;

  if (!impl_->mean.empty() && impl_->feature_dim == feature_dim) {
    normalized_features.resize(static_cast<size_t>(num_samples) * feature_dim);
    for (int i = 0; i < num_samples; ++i) {
      for (int j = 0; j < feature_dim; ++j) {
        normalized_features[static_cast<size_t>(i) * feature_dim + j] =
            static_cast<float>(
                (features[static_cast<size_t>(i) * feature_dim + j] -
                 impl_->mean[j]) /
                impl_->std_dev[j]);
      }
    }
    features_to_use = normalized_features.data();
  }

  // Convert dense features to sparse node2d format (ThunderSVM requirement)
  DataSet::node2d instances(num_samples);

  for (int i = 0; i < num_samples; ++i) {
    instances[i].reserve(feature_dim);
    const float *sample =
        features_to_use + static_cast<size_t>(i) * feature_dim;

    for (int j = 0; j < feature_dim; ++j) {
      if (sample[j] != 0.0f) { // Only non-zero values (sparse format)
        instances[i].emplace_back(j + 1, sample[j]); // 1-based index
      }
    }
  }

  // Predict using ThunderSVM (uses GPU internally)
  std::vector<double> preds = impl_->model->predict(instances, num_samples);

  // Convert to int labels
  for (int i = 0; i < num_samples; ++i) {
    predictions[i] = static_cast<int>(preds[i]);
  }

  return predictions;
}

float SVMWrapper::evaluate(const float *features, const int *labels,
                           int num_samples, int feature_dim) const {
  std::vector<int> predictions = predict(features, num_samples, feature_dim);

  int correct = 0;
  for (int i = 0; i < num_samples; ++i) {
    if (predictions[i] == labels[i]) {
      ++correct;
    }
  }

  return static_cast<float>(correct) / num_samples;
}

bool SVMWrapper::save_model(const std::string &path) const {
  if (!impl_->model)
    return false;
  impl_->model->save_to_file(path);
  return true;
}

bool SVMWrapper::load_model(const std::string &path) {
  impl_->model = std::make_unique<SVC>();
  impl_->model->load_from_file(path);
  return true;
}

// ==============================================================================
// LIBSVM Implementation (CPU fallback)
// ==============================================================================
#elif defined(WITH_LIBSVM)

#include "svm.h" // LIBSVM header

struct SVMWrapper::Impl {
  svm_model *model = nullptr;
  svm_parameter param;
  double gamma_value = 0.0;

  Impl() {
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0;
    param.coef0 = 0;
    param.C = 10.0;
    param.nu = 0.5;
    param.eps = 1e-3;
    param.cache_size = 200;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;
  }

  ~Impl() {
    if (model) {
      svm_free_and_destroy_model(&model);
    }
  }
};

SVMWrapper::SVMWrapper() : impl_(std::make_unique<Impl>()) {}

SVMWrapper::~SVMWrapper() = default;

void SVMWrapper::set_kernel(int kernel_type) {
  impl_->param.kernel_type = kernel_type;
}

void SVMWrapper::set_C(double C) { impl_->param.C = C; }

void SVMWrapper::set_gamma(double gamma) { impl_->gamma_value = gamma; }

void SVMWrapper::train(const float *features, const int *labels,
                       int num_samples, int feature_dim) {
  std::cout << "Training LIBSVM (CPU) on " << num_samples << " samples, "
            << feature_dim << " features..." << std::endl;

  if (impl_->gamma_value <= 0) {
    impl_->param.gamma = 1.0 / feature_dim;
  } else {
    impl_->param.gamma = impl_->gamma_value;
  }

  svm_problem prob;
  prob.l = num_samples;
  prob.y = new double[num_samples];
  prob.x = new svm_node *[num_samples];

  for (int i = 0; i < num_samples; ++i) {
    prob.y[i] = static_cast<double>(labels[i]);
    prob.x[i] = new svm_node[feature_dim + 1];

    const float *sample = features + static_cast<size_t>(i) * feature_dim;
    for (int j = 0; j < feature_dim; ++j) {
      prob.x[i][j].index = j + 1;
      prob.x[i][j].value = static_cast<double>(sample[j]);
    }
    prob.x[i][feature_dim].index = -1;
  }

  const char *error_msg = svm_check_parameter(&prob, &impl_->param);
  if (error_msg) {
    std::cerr << "SVM parameter error: " << error_msg << std::endl;
    for (int i = 0; i < num_samples; ++i)
      delete[] prob.x[i];
    delete[] prob.x;
    delete[] prob.y;
    return;
  }

  impl_->model = svm_train(&prob, &impl_->param);

  if (impl_->model) {
    std::cout << "LIBSVM training complete. Support vectors: "
              << impl_->model->l << std::endl;
  }

  for (int i = 0; i < num_samples; ++i) {
    delete[] prob.x[i];
  }
  delete[] prob.x;
  delete[] prob.y;
}

std::vector<int> SVMWrapper::predict(const float *features, int num_samples,
                                     int feature_dim) const {
  std::vector<int> predictions(num_samples, -1);

  if (!impl_->model) {
    std::cerr << "Error: SVM model not trained!" << std::endl;
    return predictions;
  }

  std::vector<svm_node> x(feature_dim + 1);

  for (int i = 0; i < num_samples; ++i) {
    const float *sample = features + static_cast<size_t>(i) * feature_dim;

    for (int j = 0; j < feature_dim; ++j) {
      x[j].index = j + 1;
      x[j].value = static_cast<double>(sample[j]);
    }
    x[feature_dim].index = -1;

    double pred = svm_predict(impl_->model, x.data());
    predictions[i] = static_cast<int>(pred);
  }

  return predictions;
}

float SVMWrapper::evaluate(const float *features, const int *labels,
                           int num_samples, int feature_dim) const {
  std::vector<int> predictions = predict(features, num_samples, feature_dim);

  int correct = 0;
  for (int i = 0; i < num_samples; ++i) {
    if (predictions[i] == labels[i]) {
      ++correct;
    }
  }

  return static_cast<float>(correct) / num_samples;
}

bool SVMWrapper::save_model(const std::string &path) const {
  if (!impl_->model)
    return false;
  return svm_save_model(path.c_str(), impl_->model) == 0;
}

bool SVMWrapper::load_model(const std::string &path) {
  if (impl_->model) {
    svm_free_and_destroy_model(&impl_->model);
  }
  impl_->model = svm_load_model(path.c_str());
  return impl_->model != nullptr;
}

// ==============================================================================
// Stub Implementation (No SVM library)
// ==============================================================================
#else

struct SVMWrapper::Impl {
  double C = 10.0;
  double gamma = 0.0;
  int kernel_type = 2;
  bool trained = false;

  std::vector<std::vector<float>> class_centroids;
  int num_classes = 10;
  int feature_dim = 0;
};

SVMWrapper::SVMWrapper() : impl_(std::make_unique<Impl>()) {}

SVMWrapper::~SVMWrapper() = default;

void SVMWrapper::set_kernel(int kernel_type) {
  impl_->kernel_type = kernel_type;
}

void SVMWrapper::set_C(double C) { impl_->C = C; }

void SVMWrapper::set_gamma(double gamma) { impl_->gamma = gamma; }

void SVMWrapper::train(const float *features, const int *labels,
                       int num_samples, int feature_dim) {
  std::cout << "[STUB] SVM training (no SVM library available)" << std::endl;
  std::cout << "  Using nearest-centroid classifier as fallback" << std::endl;

  impl_->feature_dim = feature_dim;
  impl_->class_centroids.resize(impl_->num_classes);

  std::vector<int> class_counts(impl_->num_classes, 0);
  for (int c = 0; c < impl_->num_classes; ++c) {
    impl_->class_centroids[c].resize(feature_dim, 0.0f);
  }

  for (int i = 0; i < num_samples; ++i) {
    int label = labels[i];
    if (label >= 0 && label < impl_->num_classes) {
      const float *sample = features + static_cast<size_t>(i) * feature_dim;
      for (int j = 0; j < feature_dim; ++j) {
        impl_->class_centroids[label][j] += sample[j];
      }
      class_counts[label]++;
    }
  }

  for (int c = 0; c < impl_->num_classes; ++c) {
    if (class_counts[c] > 0) {
      for (int j = 0; j < feature_dim; ++j) {
        impl_->class_centroids[c][j] /= class_counts[c];
      }
    }
  }

  impl_->trained = true;
  std::cout << "  Training complete (stub)" << std::endl;
}

std::vector<int> SVMWrapper::predict(const float *features, int num_samples,
                                     int feature_dim) const {
  std::vector<int> predictions(num_samples, -1);

  if (!impl_->trained) {
    std::cerr << "Error: Model not trained!" << std::endl;
    return predictions;
  }

  for (int i = 0; i < num_samples; ++i) {
    const float *sample = features + static_cast<size_t>(i) * feature_dim;

    float min_dist = 1e30f;
    int best_class = 0;

    for (int c = 0; c < impl_->num_classes; ++c) {
      float dist = 0.0f;
      for (int j = 0; j < feature_dim; ++j) {
        float d = sample[j] - impl_->class_centroids[c][j];
        dist += d * d;
      }
      if (dist < min_dist) {
        min_dist = dist;
        best_class = c;
      }
    }

    predictions[i] = best_class;
  }

  return predictions;
}

float SVMWrapper::evaluate(const float *features, const int *labels,
                           int num_samples, int feature_dim) const {
  std::vector<int> predictions = predict(features, num_samples, feature_dim);

  int correct = 0;
  for (int i = 0; i < num_samples; ++i) {
    if (predictions[i] == labels[i]) {
      ++correct;
    }
  }

  return static_cast<float>(correct) / num_samples;
}

bool SVMWrapper::save_model(const std::string & /*path*/) const {
  std::cerr << "[STUB] save_model not implemented" << std::endl;
  return false;
}

bool SVMWrapper::load_model(const std::string & /*path*/) {
  std::cerr << "[STUB] load_model not implemented" << std::endl;
  return false;
}

#endif // WITH_THUNDERSVM / WITH_LIBSVM

// ==============================================================================
// Utility Functions
// ==============================================================================
void print_confusion_matrix(const std::vector<int> &true_labels,
                            const std::vector<int> &pred_labels,
                            int num_classes) {
  if (true_labels.size() != pred_labels.size()) {
    std::cerr << "Error: label vectors have different sizes" << std::endl;
    return;
  }

  std::vector<std::vector<int>> matrix(num_classes,
                                       std::vector<int>(num_classes, 0));

  for (size_t i = 0; i < true_labels.size(); ++i) {
    int t = true_labels[i];
    int p = pred_labels[i];
    if (t >= 0 && t < num_classes && p >= 0 && p < num_classes) {
      matrix[t][p]++;
    }
  }

  std::cout << "\nConfusion Matrix:\n";
  std::cout << "True\\Pred ";
  for (int j = 0; j < num_classes; ++j) {
    std::cout << std::setw(5) << j;
  }
  std::cout << "\n";

  for (int i = 0; i < num_classes; ++i) {
    std::cout << std::setw(9) << i << " ";
    for (int j = 0; j < num_classes; ++j) {
      std::cout << std::setw(5) << matrix[i][j];
    }
    std::cout << "\n";
  }

  std::cout << "\nPer-class accuracy:\n";
  for (int i = 0; i < num_classes; ++i) {
    int total = 0;
    for (int j = 0; j < num_classes; ++j) {
      total += matrix[i][j];
    }
    float acc = (total > 0) ? static_cast<float>(matrix[i][i]) / total : 0.0f;
    std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(1)
              << (acc * 100) << "%" << std::endl;
  }
}
