#include <iostream>
#include <cmath>
#include <algorithm>
#include <utility>
#include <numeric>
#include <random>
#include <eigen3/Eigen/Dense>

#define LOG_PROGRESS
#define MAX_ITERS 10000
#define TOLERANCE 0

using Eigen::VectorXd;
using Eigen::MatrixXd;

double sign(double n) {
	if (n < 0) return -1;
	else if (n > 0) return 1;
	else return 0;
}

VectorXd sign(const VectorXd& v) {
	return v.array().sign();
}

VectorXd gd_ridge(
	const MatrixXd& X_train, const MatrixXd& X_valid, 
	const VectorXd& y_train, const VectorXd& y_valid, 
	double lam, double tau
) {
	int m = X_train.rows(), n = X_train.cols();
	VectorXd w = VectorXd::Zero(n);
	double min_loss = std::numeric_limits<double>::infinity();
	for (int k = 0; k < MAX_ITERS; k++) {
		w -= tau * X_train.transpose() * (X_train * w - y_train) - tau * lam * w;
		
		double loss = (X_valid * w - y_valid).squaredNorm() + lam * w.squaredNorm();
		if (loss < min_loss) {
			min_loss = loss;
		} else if (loss > min_loss + TOLERANCE) {
			#ifdef LOG_PROGRESS
			std::cout << "stopped early\n";
			#endif
			break;
		}

		#ifdef LOG_PROGRESS
		if (k % 100 == 0) std::cout << 100.0 * k / MAX_ITERS << "%\n";
		#endif
	}	
	return w;
}

VectorXd gd_lasso(
	const MatrixXd& X_train, const MatrixXd& X_valid, 
	const VectorXd& y_train, const VectorXd& y_valid, 
	double lam, double tau
) {
	int m = X_train.rows(), n = X_train.cols();
	VectorXd w = VectorXd::Zero(n);
	double min_loss = std::numeric_limits<double>::infinity();
	for (int k = 0; k < MAX_ITERS; k++) {
		w -= tau * X_train.transpose() * (X_train * w - y_train) - 0.5 * tau * lam * sign(w);
		
		double loss = (X_valid * w - y_valid).squaredNorm() + lam * w.lpNorm<1>();
		if (loss < min_loss) {
			min_loss = loss;
		} else if (loss > min_loss + TOLERANCE) {
			#ifdef LOG_PROGRESS
			std::cout << "stopped early\n";
			#endif
			break;
		}

		#ifdef LOG_PROGRESS
		if (k % 100 == 0) std::cout << 100.0 * k / MAX_ITERS << "%\n";
		#endif
	}	
	return w;
}

VectorXd gd_svm(
	const MatrixXd& X_train, const MatrixXd& X_valid, 
	const VectorXd& y_train, const VectorXd& y_valid, 
	double lam, double tau
) {
	int m = X_train.rows(), n = X_train.cols();
	VectorXd w = VectorXd::Zero(n);
	double min_loss = std::numeric_limits<double>::infinity();
	for (int k = 0; k < MAX_ITERS; k++) {
		VectorXd w_new = (1.0 - lam * tau) * w;
		for (int i = 0; i < m; i++) {
			if (y_train(i) * X_train.row(i) * w < 1) {
				w_new += 0.5 * tau * y_train(i) * X_train.row(i);
			}
		}
		w = w_new;
		
		double loss = lam * w.squaredNorm();
		if (loss < min_loss) {
			min_loss = loss;
		} else if (loss > min_loss + TOLERANCE) {
			#ifdef LOG_PROGRESS
			std::cout << "stopped early\n";
			#endif
			break;
		}

		#ifdef LOG_PROGRESS
		if (k % 100 == 0) std::cout << 100.0 * k / MAX_ITERS << "%\n";
		#endif
	}	
	return w;
}