#include <LinearRegression.hpp>

#define MAX_ITERS 1e6

using namespace mm;

void LinearRegression::fit(const MatrixXd& X, const VectorXd& y) {
    w = (X.transpose() * X).inverse() * X.transpose() * y;
}

void LinearRegression::fit(const MatrixXd& X, const VectorXd& y, int num_iters, double lr) {
    w = VectorXd::Random(X.cols());
    for (int i = 0; i < num_iters; i++) {
        w -= lr * X.transpose() * (X * w - y);
    }
}

void LinearRegression::fit(const MatrixXd& X, const VectorXd& y, double tol, double lr) {
    w = VectorXd::Random(X.cols());
    for (int i = 0; i < MAX_ITERS; i++) {
        VectorXd grad =  X.transpose() * (X * w - y);
        w -= lr * grad;
        if (grad.norm() < tol) break;
    }
}

VectorXd LinearRegression::predict(const MatrixXd& X) {
    return X * w;
}

void LinearRegressionL1::fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr) {
    w = VectorXd::Random(X.cols());
    for (int i = 0; i < num_iters; i++) {
        VectorXd w_sign = w.array().sign();
        w -= lr * (X.transpose() * (X * w - y) + lam * w_sign);
    }
}

void LinearRegressionL1::fit(const MatrixXd& X, const VectorXd& y, double lam, double tol, double lr) {
    w = VectorXd::Random(X.cols());
    for (int i = 0; i < MAX_ITERS; i++) {
        VectorXd w_sign = w.array().sign();
        VectorXd grad = X.transpose() * (X * w - y) + lam * w_sign;
        w -= lr * grad;
        if (grad.norm() < tol) break;
    }
}

VectorXd LinearRegressionL1::predict(const MatrixXd& X) {
    return X * w;
}

void LinearRegressionL2::fit(const MatrixXd& X, const VectorXd& y, double lam) {
    w = (X.transpose() * X + lam * MatrixXd::Identity(X.cols(), X.cols())).inverse() * X.transpose() * y;
}

void LinearRegressionL2::fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr) {
    w = VectorXd::Random(X.cols());
    for (int i = 0; i < num_iters; i++) {
        w -= lr * (X.transpose() * (X * w - y) + lam * w);
    }
}

void LinearRegressionL2::fit(const MatrixXd& X, const VectorXd& y, double lam, double tol, double lr) {
    w = VectorXd::Random(X.cols());
    for (int i = 0; i < MAX_ITERS; i++) {
        VectorXd grad = X.transpose() * (X * w - y) + lam * w;
        w -= lr * grad;
        if (grad.norm() < tol) break;
    }
}

VectorXd LinearRegressionL2::predict(const MatrixXd& X) {
    return X * w;
}