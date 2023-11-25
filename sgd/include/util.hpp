#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <eigen3/Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

template<typename Fn, typename... Args>
auto time_fn(Fn&& fn, Args&&... args) {
	auto start = std::chrono::high_resolution_clock::now();
	auto result = std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	double seconds = duration.count();
	return std::make_pair(result, seconds);
}

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

MatrixXd csv2matrix_with_ones(const std::string& path) {
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
	MatrixXd X = MatrixXd::Ones(m, n + 1);
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

MatrixXd csv2matrix_with_squares_and_ones(const std::string& path) {
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
	MatrixXd X = MatrixXd::Ones(m, 2 * n + 1);
	for (int i = 0; std::getline(file, line); i++) {
		std::istringstream ss(line);
		std::string cell;
		for (int j = 0; std::getline(ss, cell, ','); j++) {
			double num = std::stod(cell);
			X(i, 2 * j) = num;
			X(i, 2 * j + 1) = num * num;
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
