#include <iostream>
#include <cmath>
#include <algorithm>
#include <utility>
#include <numeric>
#include <random>
#include <eigen3/Eigen/Dense>

#define NUM_ITERS 10000
#define MAX_ITERS 10000
#define TOLERANCE 1e-2

using Eigen::VectorXd;
using Eigen::MatrixXd;

double sign(double n) {
	if (n < 0) return -1;
	else if (n > 0) return 1;
	else return 0;
}

VectorXd sign(const VectorXd& v) {
	int n = v.size();
	VectorXd res(n);
	for (int i = 0; i < n; i++) {
		res(i) = sign(v(i));
	}
	return res;
}

VectorXd sgd_ridge(const MatrixXd& X, const VectorXd& y, double lam, double tau) {
	int m = X.rows(), n = X.cols();
	std::random_device rd;
  std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(0, m - 1);
	// Eigen::Vector<size_t, Eigen::Dynamic> rand_ints = Eigen::Vector<size_t, Eigen::Dynamic>::Random(m * NUM_ITERS);
	VectorXd w = VectorXd::Zero(n);
	for (int k = 0; k < m * NUM_ITERS; k++) {
		// int i = rand_ints(k) % m;
		int i = dist(gen);
		w -= tau * (X.row(i) * w - y(i)) * X.row(i).transpose() + (tau * lam / m) * w;
	}	
	return w;
}

VectorXd sgd_lasso(const MatrixXd& X, const VectorXd& y, double lam, double tau) {
	int m = X.rows(), n = X.cols();
	std::random_device rd;
  	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(0, m - 1);
	VectorXd w = VectorXd::Zero(n);
	for (int k = 0; k < m * NUM_ITERS; k++) {
		int i = dist(gen);
		w -= tau * (X.row(i) * w - y(i)) * X.row(i).transpose() + (0.5 * tau * lam / m) * sign(w);
	}	
	return w;
}

VectorXd sgd_svm(const MatrixXd& X, const VectorXd& y, double lam, double tau) {
	int m = X.rows(), n = X.cols();
	std::random_device rd;
  	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(0, m - 1);
	VectorXd w = VectorXd::Zero(n);
	for (int k = 0; k < m * NUM_ITERS; k++) {
		int i = dist(gen);
		if (y(i) * X.row(i) * w < 1) {
			w += 0.5 * tau * y(i) * X.row(i).transpose() - (tau * lam / m) * w;
		} else {
			w -= (tau * lam / m) * w;
		}
	}	
	return w;
}

VectorXd gd_ridge(const MatrixXd& X, const VectorXd& y, double lam, double tau) {
	int m = X.rows(), n = X.cols();
	VectorXd w = VectorXd::Zero(n);
	for (int k = 0; k < NUM_ITERS; k++) {
		w -= tau * X.transpose() * (X * w - y) - tau * lam * w;
	}	
	return w;
}

VectorXd gd_lasso(const MatrixXd& X, const VectorXd& y, double lam, double tau) {
	int m = X.rows(), n = X.cols();
	VectorXd w = VectorXd::Zero(n);
	for (int k = 0; k < NUM_ITERS; k++) {
		w -= tau * X.transpose() * (X * w - y) - 0.5 * tau * lam * sign(w);
	}	
	return w;
}

VectorXd gd_svm(const MatrixXd& X, const VectorXd& y, double lam, double tau) {
	int m = X.rows(), n = X.cols();
	VectorXd w = VectorXd::Zero(n);
	for (int k = 0; k < NUM_ITERS; k++) {
		VectorXd w_new = (1.0 - lam * tau) * w;
		for (int i = 0; i < m; i++) {
			if (y(i) * X.row(i) * w < 1) {
				w_new += 0.5 * tau * y(i) * X.row(i);
			}
		}
		w = w_new;
	}	
	return w;
}

VectorXd sgd_ridge_early_stop(
	const MatrixXd& X_train, const MatrixXd& X_valid, 
	const VectorXd& y_train, const VectorXd& y_valid, 
	double lam, double tau
) {
	int m = X_train.rows(), n = X_train.cols();
	std::random_device rd;
  std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(0, m - 1);
	VectorXd w = VectorXd::Zero(n);
	double min_loss = std::numeric_limits<double>::infinity();
	for (int k = 0; k < m * MAX_ITERS; k++) {
		int i = dist(gen);
		w -= tau * (X_train.row(i) * w - y_train(i)) * X_train.row(i).transpose() + (tau * lam / m) * w;
		if (k % m == 0) {
			double loss = (X_valid * w - y_valid).squaredNorm() + lam * w.squaredNorm();
			if (loss < min_loss) min_loss = loss;
			else if (loss > min_loss + TOLERANCE) break;
		}
	}	
	return w;
}

VectorXd sgd_lasso_early_stop(
	const MatrixXd& X_train, const MatrixXd& X_valid, 
	const VectorXd& y_train, const VectorXd& y_valid, 
	double lam, double tau
) {
	int m = X_train.rows(), n = X_train.cols();
	std::random_device rd;
  std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(0, m - 1);
	VectorXd w = VectorXd::Zero(n);
	double min_loss = std::numeric_limits<double>::infinity();
	for (int k = 0; k < m * MAX_ITERS; k++) {
		int i = dist(gen);
		w -= tau * (X_train.row(i) * w - y_train(i)) * X_train.row(i).transpose() + (0.5 * tau * lam / m) * sign(w);
		if (k % m == 0) {
			double loss = (X_valid * w - y_valid).squaredNorm() + lam * w.lpNorm<1>();
			if (loss < min_loss) min_loss = loss;
			else if (loss > min_loss + TOLERANCE) break;
		}
	}	
	return w;
}

VectorXd sgd_svm_early_stop(
	const MatrixXd& X_train, const MatrixXd& X_valid, 
	const VectorXd& y_train, const VectorXd& y_valid, 
	double lam, double tau
) {
	int m = X_train.rows(), n = X_train.cols();
	std::random_device rd;
  std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(0, m - 1);
	VectorXd w = VectorXd::Zero(n);
	double min_loss = std::numeric_limits<double>::infinity();
	for (int k = 0; k < m * MAX_ITERS; k++) {
		int i = dist(gen);
		if (y_train(i) * X_train.row(i) * w < 1) {
			w += 0.5 * tau * y_train(i) * X_train.row(i).transpose() - (tau * lam / m) * w;
		} else {
			w -= (tau * lam / m) * w;
		}
		if (k % m == 0) {
			double loss = lam * w.squaredNorm();
			for (int i = 0; i < y_valid.size(); i++) {
				loss += std::max(1.0 - y_valid(i) * X_valid.row(i) * w, 0.0);
			}
			if (loss < min_loss) min_loss = loss;
			else if (loss > min_loss + TOLERANCE) break;
		}
	}	
	return w;
}

VectorXd gd_ridge_early_stop(
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
		if (loss < min_loss) min_loss = loss;
		else if (loss > min_loss + TOLERANCE) break;
	}	
	return w;
}

VectorXd gd_lasso_early_stop(
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
		if (loss < min_loss) min_loss = loss;
		else if (loss > min_loss + TOLERANCE) break;
	}	
	return w;
}

VectorXd gd_svm_early_stop(
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
		for (int i = 0; i < y_valid.size(); i++) {
			loss += std::max(1.0 - y_valid(i) * X_valid.row(i) * w, 0.0);
		}
		if (loss < min_loss) min_loss = loss;
		else if (loss > min_loss + TOLERANCE) break;
	}	
	return w;
}