/**
 * @file svm_wrapper.h
 * @brief SVM Classifier Wrapper
 *
 * Unified interface for SVM classification supporting multiple backends:
 * - ThunderSVM (GPU-accelerated, compile with WITH_THUNDERSVM)
 * - LIBSVM (CPU, compile with WITH_LIBSVM)
 * - Stub (no-op, when neither is available)
 *
 * Uses pimpl idiom for backend-agnostic header.
 */

#ifndef SVM_WRAPPER_H
#define SVM_WRAPPER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief SVM Classifier Wrapper
 *
 * Provides unified interface for training and prediction.
 * Backend is selected at compile time via preprocessor macros.
 *
 * Usage:
 *   SVMWrapper svm;
 *   svm.train(features, labels, num_samples, feature_dim);
 *   auto predictions = svm.predict(test_features, num_test, feature_dim);
 *   float accuracy = svm.evaluate(test_features, test_labels, num_test,
 * feature_dim);
 */
class SVMWrapper {
public:
  SVMWrapper();
  ~SVMWrapper();

  /// Train SVM on feature vectors
  void train(const float *features, const int *labels, int num_samples,
             int feature_dim);

  /// Predict labels for feature vectors
  std::vector<int> predict(const float *features, int num_samples,
                           int feature_dim) const;

  /// Compute accuracy on test set
  float evaluate(const float *features, const int *labels, int num_samples,
                 int feature_dim) const;

  /// Save trained model to file
  bool save_model(const std::string &path) const;

  /// Load model from file
  bool load_model(const std::string &path);

  // ===== Hyperparameter Setters =====
  void set_kernel(int kernel_type); ///< 0=LINEAR, 2=RBF
  void set_C(double C);             ///< Regularization parameter
  void set_gamma(double gamma);     ///< RBF kernel parameter

private:
  struct Impl; ///< pimpl for backend-specific implementation
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Print confusion matrix to stdout
 * @param num_classes Number of classes (10 for CIFAR-10)
 */
void print_confusion_matrix(const std::vector<int> &true_labels,
                            const std::vector<int> &pred_labels,
                            int num_classes);

#endif // SVM_WRAPPER_H
