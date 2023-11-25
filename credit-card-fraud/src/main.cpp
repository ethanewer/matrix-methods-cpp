#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <eigen3/Eigen/Dense>
#include <gd.hpp>

#define LOG(x) std::cout << #x << ": " << (x) << '\n'

using Eigen::VectorXd;
using Eigen::MatrixXd;

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> split_data(const MatrixXd& X, const VectorXd& y, double a, double b) {
	int m = X.rows(), n = X.cols();
	int i = std::round(a * m / (a + b));
	return {X.block(0, 0, i, n), X.block(i, 0, m - i, n), y.segment(0, i), y.segment(i, m - i)};
}

double test_classifier(const MatrixXd& X, const VectorXd& y, const VectorXd& w) {
	VectorXd preds = X * w;
	double error_count = 0.0;
	for (int k = 0; k < y.size(); k++) {
		if (sign(preds(k)) != y(k)) error_count += 1;
	}
	return error_count / y.size();
}

int main() {
	std::ifstream file("../../data/credit-card-fraud/creditcard_2023.csv");
	std::string line;
	std::getline(file, line); // skip header
	int num_false = 0, num_true = 0;
	std::vector<std::vector<double>> data;
	while (std::getline(file, line)) {
		std::istringstream ss(line);
		std::string cell;
		std::vector<double> row;
		while (std::getline(ss, cell, ',')) {
			row.push_back(std::stod(cell));
		}
		if (row.back() == 0) {
			row[row.size() - 1] = -1;
		}  
		data.push_back(row);
	}
	file.close();
	if (data.empty()) throw std::runtime_error("empty CSV file");

	std::random_device rd;
  std::mt19937 gen(rd());
	std::shuffle(data.begin(), data.end(), gen);

	int m = data.size(), n = data[0].size() - 2;
	MatrixXd X(m, n);
	VectorXd y(m);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			X(i, j) = data[i][j + 1];
		}
		y(i) = data[i].back();
	}

	for (int i = 0; i < n; i++) {
		double mean = X.col(i).mean();
		double variance = sqrt((X.col(i).array() - mean).square().sum());
		X.col(i) = (X.col(i).array() - mean) / variance;
	}

	auto [X_train, X_valid, y_train, y_valid] = split_data(X, y, 0.5, 0.5);
	double tau = 1e-2;
	double lam = 1.0;

	VectorXd w = gd_lasso(X_train, X_valid, y_train, y_valid, lam, tau);

	LOG(test_classifier(X_valid, y_valid, w)); 
}
