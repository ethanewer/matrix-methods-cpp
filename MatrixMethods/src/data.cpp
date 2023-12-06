#include <data.hpp>


VectorXd mm::csv2vector(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open file '" + path + "'");
	}
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

VectorXd mm::csv2vector(const std::string& path, int max_size) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open file '" + path + "'");
	}
	std::string line;
	int m = 0;
	while (m < max_size && std::getline(file, line)) {
		m++;
	}
	file.close();
	file.open(path);
	VectorXd x(m);
	for (int i = 0; i < m; i++) {
		std::getline(file, line);
		x(i) = stod(line);
	}
	file.close();
	return x;
}

MatrixXd mm::csv2matrix(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open file '" + path + "'");
	}
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

MatrixXd mm::csv2matrix(const std::string& path, int max_size) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open file '" + path + "'");
	}
	std::string line;
	int m = 0, n = 0;
	while (m < max_size && std::getline(file, line)) {
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
	for (int i = 0; i < m; i++) {
		std::getline(file, line);
		std::istringstream ss(line);
		std::string cell;
		for (int j = 0; std::getline(ss, cell, ','); j++) {
			X(i, j) = std::stod(cell);
		}
	}
	file.close();
	return X;
}

MatrixXd mm::csv2matrix_with_ones(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open file '" + path + "'");
	}
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

MatrixXd mm::normalize(const MatrixXd& X) {
	int m = X.rows(), n = X.cols();
	MatrixXd X_normalized(m, n);
	for (int i = 0; i < n; i++) {
		double mean = X.col(i).mean();
		double variance = sqrt((X.col(i).array() - mean).square().sum());
		X_normalized.col(i) = (X.col(i).array() - mean) / variance;
	}
	return X_normalized;
}

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> mm::split_data(const MatrixXd& X, const VectorXd& y, double a, double b) {
	int m = X.rows(), n = X.cols();
	int i = std::round(a * m / (a + b));
	return {X.block(0, 0, i, n), X.block(i, 0, m - i, n), y.segment(0, i), y.segment(i, m - i)};
}
