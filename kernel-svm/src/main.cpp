#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <util.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

double K(const VectorXd& u, const VectorXd& v, double sig) {
	return exp(-(u - v).squaredNorm() / (2 * sig * sig));
}

double test_binary_classifier(const VectorXd& alpha, const MatrixXd& X_train, const MatrixXd& X_test, const VectorXd& y) {
	int m_test = X_test.rows(), m_train = X_train.rows();
	double error_count = 0;
	for (int i = 0; i < m_test; i++) {
		double pred = 0;
		for (int j = 0; j < m_train; j++) {
			pred += alpha(j) * K(X_test.row(i), X_train.row(j), 0.1);
		}
		pred = pred < 0 ? -1 : 1;

		if (pred != y(i)) {
			error_count++;
		}
	}
	return error_count / m_test;
}

int main() {
	auto [X_train, X_test, y_train, y_test] = load_gd_data();
	int m = X_train.rows(), n = X_train.cols();

	VectorXd alpha = VectorXd::Random(m);
	double lam = 0.01, sig = 0.1, tau = 0.01;

	for (int epoch = 0; epoch < 100; epoch++) {
		VectorXd pred = VectorXd::Zero(m);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				pred(i) += alpha(j) * K(X_train.row(i), X_train.row(j), sig);
			}
		}

		VectorXd grad = lam * pred;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				if (y_train(j) * pred(j) < 0) grad(i) -= y_train(j) * K(X_train.row(i), X_train.row(j), sig);
			}
		}

		alpha -= tau * grad;

		if (epoch < 10 || (epoch + 1) % 10 == 0) {
			std::cout << "[epoch " << epoch + 1 << "]";
			std::cout << " train error rate: " << 100 * test_binary_classifier(alpha, X_train, X_train, y_train) << "%,";
			std::cout << " test error rate: " << 100 * test_binary_classifier(alpha, X_train, X_test, y_test) << "%";
			std::cout << std::endl;
		}
	}
}
