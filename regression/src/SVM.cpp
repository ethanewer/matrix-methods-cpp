#include <SVM.hpp>

#define MAX_ITERS 1e6

void SVM::fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr) {
    w = VectorXd::Random(X.cols());
    for (int i = 0; i < num_iters; i++) {
        VectorXd grad = lam * w;
        for (int j = 0; j < X.rows(); j++) {
			if (y(j) * X.row(j) * w < 1.0) {
				grad -= 0.5 * y(j) * X.row(j);
			}
		}
        w -= lr * grad;
    }
}

void SVM::fit(const MatrixXd& X, const VectorXd& y, double lam, double tol, double lr) {
    w = VectorXd::Random(X.cols());
    for (int i = 0; i < MAX_ITERS; i++) {
        VectorXd grad = lam * w;
        for (int j = 0; j < X.rows(); j++) {
			if (y(j) * X.row(j) * w < 1.0) {
				grad -= 0.5 * y(j) * X.row(j);
			}
		}
        w -= lr * grad;
        if (grad.norm() < tol) break;
    }
}

VectorXd SVM::predict(const MatrixXd& X) {
    return X * w;
}