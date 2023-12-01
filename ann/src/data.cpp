#include <data.hpp>

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

MatrixXd normalize(const MatrixXd& X) {
	int m = X.rows(), n = X.cols();
	MatrixXd X_normalized(m, n);
	for (int i = 0; i < n; i++) {
		double mean = X.col(i).mean();
		double variance = sqrt((X.col(i).array() - mean).square().sum());
		X_normalized.col(i) = (X.col(i).array() - mean) / variance;
	}
	return X_normalized;
}

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> split_data(const MatrixXd& X, const VectorXd& y, double a, double b) {
	int m = X.rows(), n = X.cols();
	int i = std::round(a * m / (a + b));
	return {X.block(0, 0, i, n), X.block(i, 0, m - i, n), y.segment(0, i), y.segment(i, m - i)};
}

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> load_credit_card_fraud_data() {
	std::ifstream file("../../data/credit-card-fraud/creditcard_2023.csv");
	std::string line;
	std::getline(file, line); // skip header
	std::vector<std::vector<double>> data;
	while (std::getline(file, line)) {
		std::istringstream ss(line);
		std::string cell;
		std::vector<double> row;
		while (std::getline(ss, cell, ',')) {
			row.push_back(std::stod(cell));
		}
		data.push_back(row);
	}
	file.close();
	if (data.empty()) throw std::runtime_error("empty CSV file");

	std::random_device rd;
  std::mt19937 gen(rd());
	std::shuffle(data.begin(), data.end(), gen);

	int m = data.size(), n = data[0].size() - 2;
	
	MatrixXd X = MatrixXd(m, n);
	VectorXd y(m);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			X(i, j) = data[i][j + 1];
		}
		y(i) = data[i].back();
	}

	X = normalize(X);

	return split_data(X, y, 0.5, 0.5);
}

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> load_gd_data() {
	return {
		csv2matrix("../../data/gd-data/X_train.csv"),
		csv2matrix("../../data/gd-data/X_valid.csv"),
		csv2vector("../../data/gd-data/y_train.csv").array().max(0),
		csv2vector("../../data/gd-data/y_valid.csv").array().max(0)
	};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> load_mnist_digits_data() {
	MatrixXd X_train = csv2matrix("../../data/mnist-digits/X_train.csv");
	MatrixXd X_test = csv2matrix("../../data/mnist-digits/X_test.csv");
	MatrixXd Y_train = csv2matrix("../../data/mnist-digits/Y_train.csv");
	MatrixXd Y_test = csv2matrix("../../data/mnist-digits/Y_test.csv");
	
	return {X_train, X_test, Y_train, Y_test};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> load_mnist_fashion_data() {
	MatrixXd X_train = csv2matrix("../../data/mnist-fashion/X_train.csv");
	MatrixXd X_valid = csv2matrix("../../data/mnist-fashion/X_test.csv");
	MatrixXd Y_train = csv2matrix("../../data/mnist-fashion/y_train.csv");
	MatrixXd Y_valid = csv2matrix("../../data/mnist-fashion/y_test.csv");
	
	return {X_train, X_valid, Y_train, Y_valid};
}

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> load_chicken_data() {
	MatrixXd X = csv2matrix("../../data/chickens/X.csv");
	MatrixXd y = csv2matrix("../../data/chickens/y.csv");
	
	return split_data(X, y, 0.5, 0.5);
}