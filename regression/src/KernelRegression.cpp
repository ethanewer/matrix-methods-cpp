#include <KernelRegression.hpp>

#define MAX_ITERS 1e5

KernelRegressionL2::KernelRegressionL2(std::function<double(const VectorXd&, const VectorXd&)> kernel_fn) : kernel_fn(kernel_fn) {}

void KernelRegressionL2::fit(const MatrixXd& X, const VectorXd& y, double lam) {
    X_train = X;
    int m = X_train.rows();
    MatrixXd K = MatrixXd(m, m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            K(i, j) = kernel_fn(X_train.row(i), X_train.row(j));
        }
    }
    a = (K + lam * MatrixXd::Identity(m, m)).inverse() * y;
}

VectorXd KernelRegressionL2::predict(const MatrixXd& X_pred) {
    int m_train = X_train.rows(), m_pred = X_pred.rows();
    MatrixXd K = MatrixXd(m_pred, m_train);
    for (int i = 0; i < m_pred; i++) {
        for (int j = 0; j < m_train; j++) {
            K(i, j) = kernel_fn(X_pred.row(i), X_train.row(j));
        }
    }
    return K * a;
}

KernelSVM::KernelSVM(std::function<double(const VectorXd&, const VectorXd&)> kernel_fn) : kernel_fn(kernel_fn) {}

void KernelSVM::fit(const MatrixXd& X, const VectorXd& y, double lam, int num_iters, double lr) {
    X_train = X;
    int m = X_train.rows();
    MatrixXd K = MatrixXd(m, m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            K(i, j) = kernel_fn(X_train.row(i), X_train.row(j));
        }
    }


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
    MatrixXd K = MatrixXd(m, m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            K(i, j) = kernel_fn(X_train.row(i), X_train.row(j));
        }
    }

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
    int m_train = X_train.rows(), m_pred = X_pred.rows();
    MatrixXd K = MatrixXd(m_pred, m_train);
    for (int i = 0; i < m_pred; i++) {
        for (int j = 0; j < m_train; j++) {
            K(i, j) = kernel_fn(X_pred.row(i), X_train.row(j));
        }
    }
    return K * a;
}
