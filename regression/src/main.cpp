#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <utility>
#include <LinearRegression.hpp>
#include <SVM.hpp>
#include <KernelRegression.hpp>

#define SIG 0.1

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

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> split_data(const MatrixXd& X, const VectorXd& y, double a, double b) {
	int m = X.rows(), n = X.cols();
	int i = std::round(a * m / (a + b));
	return {X.block(0, 0, i, n), X.block(i, 0, m - i, n), y.segment(0, i), y.segment(i, m - i)};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> load_face_emotion_data() {
	MatrixXd X = csv2matrix("../../data/face-emotion/X.csv");
	VectorXd y = csv2vector("../../data/face-emotion/y.csv");
	return split_data(X, y, 0.8, 0.2);
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

double K(const VectorXd& u, const VectorXd& v) {
	return exp(-(u - v).squaredNorm() / (2 * SIG * SIG));
}

int main() {
    auto [X_train, X_test, y_train, y_test] = load_face_emotion_data();
	KernelRegressionL2 model(K);
    
    model.fit(X_train, y_train, 1);
	std::cout << model.predict(X_test) << "\n\n";
    std::cout << test_classifier(model.predict(X_test), y_test) << '\n';

    // model.fit(X_train, y_train, 1, 1000, 1e-3);
    // std::cout << test_classifier(model.predict(X_test), y_test) << '\n';

    // model.fit(X_train, y_train, 1, 150.0, 1e-3);
    // std::cout << test_classifier(model.predict(X_test), y_test) << '\n';
}
