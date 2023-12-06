#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <utility>
#include <LinearRegression.hpp>

VectorXd csv2vector(const std::string& path) {
	std::ifstream file(path);
	std::string line;
	int m = 0;
	while (std::getline(file, line)) {
		m++;
	}
	file.close();
	file.open(path);
	VectorXd x(m);
	for (int i = 0; std::getline(file, line); i++) {
		x(i) = stod(line);
	}
	file.close();
	return x;
}

MatrixXd csv2matrix(const std::string& path) {
	std::ifstream file(path);
	std::string line;
	int m = 0, n = 0;
	while (std::getline(file, line)) {
		m++;
		if (n == 0) {
			std::istringstream ss(line);
			std::string cell;
			while (std::getline(ss, cell, ',')) n++;
		}
	}
	file.close();
	file.open(path);
	MatrixXd X(m, n);
	for (int i = 0; std::getline(file, line); i++) {
		std::istringstream ss(line);
		std::string cell;
		for (int j = 0; std::getline(ss, cell, ','); j++) {
			X(i, j) = std::stod(cell);
		}
	}
	file.close();
	return X;
}

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> load_gd_data() {
	return {
		csv2matrix("../../data/gd-data/X_train.csv"),
		csv2matrix("../../data/gd-data/X_valid.csv"),
		csv2vector("../../data/gd-data/y_train.csv"),
		csv2vector("../../data/gd-data/y_valid.csv"),
	};
}

double sign(double n) {
	if (n < 0) return -1;
	else if (n > 0) return 1;
	else return 0;
}

double test_classifier(const MatrixXd& preds, const VectorXd& y) {
	double error_count = 0.0;
	for (int k = 0; k < y.size(); k++) {
		if (sign(preds(k)) != y(k)) error_count += 1;
	}
	return error_count / y.size();
}

int main() {
    auto [X_train, X_test, y_train, y_test] = load_gd_data();
    LinearRegressionL2 linear_regression;
    
    linear_regression.fit(X_train, y_train, 1);
    std::cout << test_classifier(linear_regression.predict(X_test), y_test) << '\n';

    linear_regression.fit(X_train, y_train, 1, 1000, 1e-3);
    std::cout << test_classifier(linear_regression.predict(X_test), y_test) << '\n';

    linear_regression.fit(X_train, y_train, 1, 1e-3, 1e-3);
    std::cout << test_classifier(linear_regression.predict(X_test), y_test) << '\n';
}
