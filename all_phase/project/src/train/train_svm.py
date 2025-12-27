#!/usr/bin/env python3
"""
train_svm.py
Train classifier on extracted autoencoder features using GPU (cuML) or CPU (sklearn)

Dataset Organization:
  - Training: 50,000 samples with extracted features (8192-dim) + labels
  - Evaluation: 10,000 test samples with extracted features + labels
  
The features were extracted by the autoencoder's encoder from CIFAR-10 images.
Classifier learns to classify these latent representations into 10 classes.

Usage:
    python train_svm.py --features cifar10_features --epochs 20
    python train_svm.py --features cifar10_features --classifier sgd  # Fastest
    python train_svm.py --features cifar10_features --classifier mlp  # Best accuracy
"""

import numpy as np
import struct
import time

def load_features(prefix):
    """Load features and labels from binary files."""
    
    def load_bin(filename, is_features=True):
        with open(filename, 'rb') as f:
            if is_features:
                num_samples = struct.unpack('i', f.read(4))[0]
                latent_dim = struct.unpack('i', f.read(4))[0]
                data = np.frombuffer(f.read(), dtype=np.float32)
                return data.reshape(num_samples, latent_dim)
            else:
                num_samples = struct.unpack('i', f.read(4))[0]
                data = np.frombuffer(f.read(), dtype=np.int32)
                return data
    
    print("Loading features...")
    X_train = load_bin(f"{prefix}_train_features.bin", is_features=True)
    y_train = load_bin(f"{prefix}_train_labels.bin", is_features=False)
    X_test = load_bin(f"{prefix}_test_features.bin", is_features=True)
    y_test = load_bin(f"{prefix}_test_labels.bin", is_features=False)
    
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, y_train, X_test, y_test


def normalize_features(X_train, X_test):
    """Normalize features to [0, 1] range using min-max scaling."""
    # Compute min/max from training set only
    X_min = X_train.min(axis=0, keepdims=True)
    X_max = X_train.max(axis=0, keepdims=True)
    
    # Avoid division by zero
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    
    X_train_norm = (X_train - X_min) / X_range
    X_test_norm = (X_test - X_min) / X_range
    
    # Clip test set to [0, 1] in case of out-of-range values
    X_test_norm = np.clip(X_test_norm, 0, 1)
    
    return X_train_norm, X_test_norm


def train_sgd_classifier(X_train, y_train, X_test, y_test, epochs=20):
    """Train SGDClassifier - very fast linear classifier with SGD optimization."""
    from sklearn.linear_model import SGDClassifier
    
    print("\nUsing SGDClassifier (CPU) - Fast linear classifier")
    print("=" * 60)
    
    best_acc = 0.0
    best_model = None
    
    # Different alpha (regularization) values
    alpha_values = np.logspace(-6, -1, epochs)
    
    for epoch in range(epochs):
        alpha = alpha_values[epoch]
        start_time = time.time()
        
        clf = SGDClassifier(
            loss='hinge',  # Linear SVM
            alpha=alpha,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        elapsed = time.time() - start_time
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = clf
            best_marker = " [BEST]"
        else:
            best_marker = ""
        
        print(f"Epoch {epoch+1:2d}/{epochs}: alpha={alpha:.2e} | "
              f"Train: {train_acc*100:5.2f}% | Test: {test_acc*100:5.2f}% | "
              f"Time: {elapsed:.2f}s{best_marker}")
    
    print("=" * 60)
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    
    return best_acc, best_model


def train_logistic_regression(X_train, y_train, X_test, y_test, epochs=20):
    """Train Logistic Regression - fast and effective for high-dim data."""
    from sklearn.linear_model import LogisticRegression
    
    print("\nUsing Logistic Regression (CPU) - Fast multiclass classifier")
    print("=" * 60)
    
    best_acc = 0.0
    best_model = None
    
    # Different C (inverse regularization) values
    C_values = np.logspace(-3, 2, epochs)
    
    for epoch in range(epochs):
        C = C_values[epoch]
        start_time = time.time()
        
        clf = LogisticRegression(
            C=C,
            solver='lbfgs',
            max_iter=500,
            multi_class='multinomial',
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        elapsed = time.time() - start_time
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = clf
            best_marker = " [BEST]"
        else:
            best_marker = ""
        
        print(f"Epoch {epoch+1:2d}/{epochs}: C={C:8.4f} | "
              f"Train: {train_acc*100:5.2f}% | Test: {test_acc*100:5.2f}% | "
              f"Time: {elapsed:.2f}s{best_marker}")
    
    print("=" * 60)
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    
    return best_acc, best_model


def train_linear_svm_fast(X_train, y_train, X_test, y_test, epochs=20):
    """Train LinearSVC with liblinear - faster than kernel SVM."""
    from sklearn.svm import LinearSVC
    
    print("\nUsing LinearSVC (CPU) - Fast linear SVM")
    print("=" * 60)
    
    best_acc = 0.0
    best_model = None
    
    C_values = np.logspace(-3, 1, epochs)
    
    for epoch in range(epochs):
        C = C_values[epoch]
        start_time = time.time()
        
        clf = LinearSVC(
            C=C,
            loss='squared_hinge',
            dual=False,  # Faster for n_samples > n_features
            max_iter=2000,
            tol=1e-4,
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        elapsed = time.time() - start_time
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = clf
            best_marker = " [BEST]"
        else:
            best_marker = ""
        
        print(f"Epoch {epoch+1:2d}/{epochs}: C={C:8.5f} | "
              f"Train: {train_acc*100:5.2f}% | Test: {test_acc*100:5.2f}% | "
              f"Time: {elapsed:.2f}s{best_marker}")
    
    print("=" * 60)
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    
    return best_acc, best_model


def train_mlp_classifier(X_train, y_train, X_test, y_test, epochs=20):
    """Train MLP classifier - neural network on top of features."""
    from sklearn.neural_network import MLPClassifier
    
    print("\nUsing MLP Classifier (CPU) - Neural network")
    print("=" * 60)
    
    best_acc = 0.0
    best_model = None
    
    # Different hidden layer sizes
    hidden_configs = [
        (256,), (512,), (1024,), (256, 128), (512, 256), 
        (1024, 512), (512, 256, 128), (256, 128, 64),
        (128,), (64,), (512, 128), (256, 64),
        (1024, 256), (512, 512), (256, 256), (128, 128),
        (2048,), (2048, 512), (1024, 256, 64), (512, 256, 128, 64)
    ]
    
    for epoch in range(min(epochs, len(hidden_configs))):
        hidden = hidden_configs[epoch]
        start_time = time.time()
        
        clf = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=256,
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        elapsed = time.time() - start_time
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = clf
            best_marker = " [BEST]"
        else:
            best_marker = ""
        
        print(f"Epoch {epoch+1:2d}/{epochs}: hidden={str(hidden):20s} | "
              f"Train: {train_acc*100:5.2f}% | Test: {test_acc*100:5.2f}% | "
              f"Time: {elapsed:.2f}s{best_marker}")
    
    print("=" * 60)
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    
    return best_acc, best_model


def train_gpu_classifier(X_train, y_train, X_test, y_test, epochs=20):
    """Train using cuML GPU-accelerated Logistic Regression (faster than LinearSVC)."""
    try:
        import cupy as cp
        from cuml.linear_model import LogisticRegression as cuLogisticRegression
        from cuml.metrics import accuracy_score
        import warnings
        warnings.filterwarnings('ignore')
        print("\nUsing cuML Logistic Regression (GPU) - Fast classifier")
    except ImportError:
        print("cuML not available, falling back to CPU SGD")
        return train_sgd_classifier(X_train, y_train, X_test, y_test, epochs)
    
    print("=" * 60)
    
    # Convert to GPU arrays
    X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
    y_train_gpu = cp.asarray(y_train, dtype=cp.int32)
    X_test_gpu = cp.asarray(X_test, dtype=cp.float32)
    y_test_gpu = cp.asarray(y_test, dtype=cp.int32)
    
    best_acc = 0.0
    best_model = None
    
    C_values = np.logspace(-2, 1, epochs)
    
    for epoch in range(epochs):
        C = C_values[epoch]
        start_time = time.time()
        
        clf = cuLogisticRegression(
            C=C,
            max_iter=200,
            tol=1e-3,
            solver='qn',  # Quasi-Newton, faster
            verbose=0
        )
        clf.fit(X_train_gpu, y_train_gpu)
        
        y_pred_train = clf.predict(X_train_gpu)
        y_pred_test = clf.predict(X_test_gpu)
        
        train_acc = accuracy_score(y_train_gpu, y_pred_train)
        test_acc = accuracy_score(y_test_gpu, y_pred_test)
        
        elapsed = time.time() - start_time
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = clf
            best_marker = " [BEST]"
        else:
            best_marker = ""
        
        print(f"Epoch {epoch+1:2d}/{epochs}: C={C:8.4f} | "
              f"Train: {train_acc*100:5.2f}% | Test: {test_acc*100:5.2f}% | "
              f"Time: {elapsed:.2f}s{best_marker}")
    
    print("=" * 60)
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    
    return best_acc, best_model


def main():
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='Train classifier on autoencoder features')
    parser.add_argument('--features', type=str, default='cifar10_features',
                        help='Prefix for feature files')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of hyperparameter iterations')
    parser.add_argument('--classifier', type=str, default='sgd', 
                        choices=['sgd', 'logistic', 'linearsvm', 'mlp', 'gpu'],
                        help='Classifier type (default: sgd - fastest)')
    parser.add_argument('--save_model', type=str, default='',
                        help='Path to save best model (pickle format)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("      Classifier Training on Autoencoder Features")
    print("=" * 60)
    
    # Load features
    X_train, y_train, X_test, y_test = load_features(args.features)
    
    # Normalize to [0, 1]
    print("\nNormalizing features to [0, 1]...")
    print(f"  Before - Train: min={X_train.min():.4f}, max={X_train.max():.4f}")
    print(f"  Before - Test:  min={X_test.min():.4f}, max={X_test.max():.4f}")
    
    X_train, X_test = normalize_features(X_train, X_test)
    
    print(f"  After  - Train: min={X_train.min():.4f}, max={X_train.max():.4f}")
    print(f"  After  - Test:  min={X_test.min():.4f}, max={X_test.max():.4f}")
    
    # Train classifier
    if args.classifier == 'sgd':
        best_acc, best_model = train_sgd_classifier(X_train, y_train, X_test, y_test, args.epochs)
    elif args.classifier == 'logistic':
        best_acc, best_model = train_logistic_regression(X_train, y_train, X_test, y_test, args.epochs)
    elif args.classifier == 'linearsvm':
        best_acc, best_model = train_linear_svm_fast(X_train, y_train, X_test, y_test, args.epochs)
    elif args.classifier == 'mlp':
        best_acc, best_model = train_mlp_classifier(X_train, y_train, X_test, y_test, args.epochs)
    elif args.classifier == 'gpu':
        best_acc, best_model = train_gpu_classifier(X_train, y_train, X_test, y_test, args.epochs)
    
    # Save model if requested
    if args.save_model and best_model is not None:
        print(f"\nSaving best model to '{args.save_model}'...")
        with open(args.save_model, 'wb') as f:
            pickle.dump(best_model, f)
        print("Model saved.")
    
    print(f"\n{'='*60}")
    print(f"Final Best Test Accuracy: {best_acc*100:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
