#include <iostream>
#include <utility>
#include <eigen3/Eigen/Dense>
#include <util.hpp>
#include <gd.hpp>

using Eigen::VectorXd;
using Eigen::MatrixXd;

double test_classifier(const MatrixXd& X, const VectorXd& y, const VectorXd& w) {
	VectorXd preds = X * w;
	double error_count = 0.0;
	for (int k = 0; k < y.size(); k++) {
		if (sign(preds(k)) != y(k)) error_count += 1;
	}
	return error_count / y.size();
}

int main() {
	MatrixXd X_train = csv2matrix_with_squares_and_ones("../../data/gd-data/X_train.csv");
	MatrixXd X_valid = csv2matrix_with_squares_and_ones("../../data/gd-data/X_valid.csv");
	VectorXd y_train = csv2vector("../../data/gd-data/y_train.csv");
	VectorXd y_valid = csv2vector("../../data/gd-data/y_valid.csv");

	// MatrixXd X = csv2matrix("../../data/breast-cancer/X.csv");
	// MatrixXd y = csv2vector("../../data/breast-cancer/y.csv");
	// auto [X_train, X_valid, y_train, y_valid] = split_data(X, y, 0.5, 0.5);

	double tau = 1.0 / pow(X_train.operatorNorm(), 2);
	VectorXd w;
	double time;

	std::tie(w, time) = time_fn(sgd_ridge, X_train, y_train, 1, tau);
	std::cout << "[stochastic gradient descent ridge] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(sgd_lasso, X_train, y_train, 1, tau);
	std::cout << "[stochastic gradient descent lasso] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(sgd_svm, X_train, y_train, 1, tau);
	std::cout << "[stochastic gradient descent svm] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(gd_ridge, X_train, y_train, 1, tau);
	std::cout << "[gradient descent ridge] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(gd_lasso, X_train, y_train, 1, tau);
	std::cout << "[gradient descent lasso] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(gd_svm, X_train, y_train, 1, tau);
	std::cout << "[gradient descent svm] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(sgd_ridge_early_stop, X_train, X_valid, y_train, y_valid, 1, tau);
	std::cout << "[stochastic gradient descent ridge (early stop)] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(sgd_lasso_early_stop, X_train, X_valid, y_train, y_valid, 1, tau);
	std::cout << "[stochastic gradient descent lasso (early stop)] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(sgd_svm_early_stop, X_train, X_valid, y_train, y_valid, 1, tau);
	std::cout << "[stochastic gradient descent svm (early stop)] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(gd_ridge_early_stop, X_train, X_valid, y_train, y_valid, 1, tau);
	std::cout << "[gradient descent ridge (early stop)] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(gd_lasso_early_stop, X_train, X_valid, y_train, y_valid, 1, tau);
	std::cout << "[gradient descent lasso (early stop)] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';

	std::tie(w, time) = time_fn(gd_svm_early_stop, X_train, X_valid, y_train, y_valid, 1, tau);
	std::cout << "[gradient descent svm (early stop)] error: ";
	std::cout << test_classifier(X_valid, y_valid, w) << ", time: " << time << '\n';
}

// dataset 1:
// [stochastic gradient descent ridge] error: 0.0535, time: 0.268716
// [stochastic gradient descent lasso] error: 0.0541, time: 0.602279
// [stochastic gradient descent svm] error: 0.0345, time: 0.259853
// [gradient descent ridge] error: 0.0547, time: 0.0383987
// [gradient descent lasso] error: 0.0542, time: 0.0387256
// [gradient descent svm] error: 0.0343, time: 0.09002
// [stochastic gradient descent ridge (early stop)] error: 0.0566, time: 0.00118446
// [stochastic gradient descent lasso (early stop)] error: 0.0574, time: 0.0028575
// [stochastic gradient descent svm (early stop)] error: 0.051, time: 0.0126063
// [gradient descent ridge (early stop)] error: 0.0543, time: 0.0190807
// [gradient descent lasso (early stop)] error: 0.0542, time: 0.129784
// [gradient descent svm (early stop)] error: 0.0393, time: 0.0665016

// dataset 2
// [stochastic gradient descent ridge] error: 0.326531, time: 20.8301
// [stochastic gradient descent lasso] error: 0.360544, time: 22.9323
// [stochastic gradient descent svm] error: 0.333333, time: 13.1872
// [gradient descent ridge] error: 0.319728, time: 2.89067
// [gradient descent lasso] error: 0.571429, time: 2.91123
// [gradient descent svm] error: 0.326531, time: 10.7273
// [stochastic gradient descent ridge (early stop)] error: 0.333333, time: 0.0118091
// [stochastic gradient descent lasso (early stop)] error: 0.319728, time: 0.00808112
// [stochastic gradient descent svm (early stop)] error: 0.312925, time: 0.0353417
// [gradient descent ridge (early stop)] error: 0.333333, time: 0.00332717
// [gradient descent lasso (early stop)] error: 0.326531, time: 0.00244508
// [gradient descent svm (early stop)] error: 0.29932, time: 0.0472231