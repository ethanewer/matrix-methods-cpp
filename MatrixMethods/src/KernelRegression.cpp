#include <KernelRegression.hpp>
#include <omp.h>

#define MAX_ITERS 1e5

using namespace mm;

KernelRegressionL2::KernelRegressionL2(std::function<double(const VectorXd&, const VectorXd&)> kernel_fn) : kernel_fn(kernel_fn) {}

void KernelRegressionL2::fit(const MatrixXd& X, const VectorXd& y, double lam) {
    X_train = X;
    int m = X_train.rows();
    MatrixXd K = make_kernel(X_train, X_train, kernel_fn);
    a = (K + lam * MatrixXd::Identity(m, m)).inverse() * y;
}

void KernelRegressionL2::fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr) {
    X_train = X;
    int m = X_train.rows();
    MatrixXd K = make_kernel(X_train, X_train, kernel_fn);
    a = VectorXd::Random(m);
    for (int i = 0; i < num_iters; i++) {
		a -= lr * (K * (K * a - y) + lam * K * a);
    }
}

void KernelRegressionL2::fit(const MatrixXd& X, const VectorXd& y, double lam, double tol, double lr) {
    X_train = X;
    int m = X_train.rows();
    MatrixXd K = make_kernel(X_train, X_train, kernel_fn);
    a = VectorXd::Random(m);
    for (int i = 0; i < MAX_ITERS; i++) {
		VectorXd grad = K * (K * a - y + lam * a);
		a -= lr * grad;
        if (grad.norm() < tol) break;
    }
}

void KernelRegressionL2::fit_with_kernel_matrix(const MatrixXd& X, const MatrixXd& K, const VectorXd& y, double lam, int num_iters, double lr) {
    X_train = X;
    int m = X_train.rows();
    a = VectorXd::Random(m);
    for (int i = 0; i < num_iters; i++) {
		a -= lr * (K * (K * a - y) + lam * K * a);
    }
}

void KernelRegressionL2::fit_with_kernel_matrix(const MatrixXd& X, const MatrixXd& K, const VectorXd& y, double lam, double tol, double lr) {
    X_train = X;
    int m = X_train.rows();
    a = VectorXd::Random(m);
    for (int i = 0; i < MAX_ITERS; i++) {
		VectorXd grad = K * (K * a - y + lam * a);
		a -= lr * grad;
        if (grad.norm() < tol) break;
    }
}

VectorXd KernelRegressionL2::predict(const MatrixXd& X_pred) {
    return make_kernel(X_pred, X_train, kernel_fn) * a;
}

VectorXd KernelRegressionL2::predict_with_kernel_matrix(const MatrixXd& K) {
    return K * a;
}

KernelSVM::KernelSVM(std::function<double(const VectorXd&, const VectorXd&)> kernel_fn) : kernel_fn(kernel_fn) {}

void KernelSVM::fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr) {
    X_train = X;
    int m = X_train.rows();
    MatrixXd K = make_kernel(X_train, X_train, kernel_fn);
    a = VectorXd::Random(m);
    for (int i = 0; i < num_iters; i++) {
        VectorXd pred = K * a;
		VectorXd grad = lam * pred;
		for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
				if (y(j) * pred(j) < 1) grad(k) -= y(j) * K(j, k);
			}
		}
		a -= lr * grad;
    }
}

void KernelSVM::fit(const MatrixXd& X, const VectorXd& y, double lam, double tol, double lr) {
    X_train = X;
    int m = X_train.rows();
    MatrixXd K = make_kernel(X_train, X_train, kernel_fn);
    a = VectorXd::Random(m);
    for (int i = 0; i < MAX_ITERS; i++) {
        VectorXd pred = K * a;
		VectorXd grad = lam * pred;
		for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
				if (y(j) * pred(j) < 1) grad(k) -= y(j) * K(j, k);
			}
		}
		a -= lr * grad;
        if (grad.norm() < tol) break;
    }
}

void KernelSVM::fit_with_kernel_matrix(const MatrixXd& X, const MatrixXd& K, const VectorXd& y, double lam, int num_iters, double lr) {
    X_train = X;
    int m = X_train.rows();
    a = VectorXd::Random(m);
    for (int i = 0; i < num_iters; i++) {
        VectorXd pred = K * a;
		VectorXd grad = lam * pred;
		for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
				if (y(j) * pred(j) < 1) grad(k) -= y(j) * K(j, k);
			}
		}
		a -= lr * grad;
    }
}

void KernelSVM::fit_with_kernel_matrix(const MatrixXd& X, const MatrixXd& K, const VectorXd& y, double lam, double tol, double lr) {
    X_train = X;
    int m = X_train.rows();
    a = VectorXd::Random(m);
    for (int i = 0; i < MAX_ITERS; i++) {
        VectorXd pred = K * a;
		VectorXd grad = lam * pred;
		for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
				if (y(j) * pred(j) < 1) grad(k) -= y(j) * K(j, k);
			}
		}
		a -= lr * grad;
        if (grad.norm() < tol) break;
    }
}

VectorXd KernelSVM::predict(const MatrixXd& X_pred) {
    return make_kernel(X_pred, X_train, kernel_fn) * a;
}

VectorXd KernelSVM::predict_with_kernel_matrix(const MatrixXd& K) {
    return K * a;
}

MatrixXd mm::make_kernel(const MatrixXd& X1, const MatrixXd& X2, std::function<double(const VectorXd&, const VectorXd&)> kernel_fn) {
    int m1 = X1.rows(), m2 = X2.rows();
    MatrixXd K = MatrixXd(m1, m2);
    #pragma omp parallel for
    for (int i = 0; i < m1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < m2; j++) {
            K(i, j) = kernel_fn(X1.row(i), X2.row(j));
        }
    }
    return K;
}