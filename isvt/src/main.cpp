#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;

MatrixXd load_csv(const std::string& path) {
	std::vector<std::vector<double>> data;
	std::ifstream file(path);
	std::string line;
	while (std::getline(file, line)) {
		std::vector<double> row;
		std::istringstream ss(line);
		std::string cell;
		while (std::getline(ss, cell, ',')) {
			row.push_back(cell == "nan" ? NAN : std::stod(cell));
		}
		data.push_back(row);
	}
	if (data.empty()) throw std::runtime_error("empty CSV file");
	
	int m = data.size(), n = data[0].size();
	MatrixXd X;
	X.resize(m, n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			X(i, j) = data[i][j];
		}
	}
	return X;
}

MatrixXd it_singular_val_thresh(MatrixXd Y, int r) {
	double tol = 1e-4;
	int m = Y.rows(), n = Y.cols();
	if (r > m || r > n) throw std::runtime_error("'r' is larger than matrix dimension.");
	MatrixXd X = Y.array().isNaN().select(0.0, Y);
	
	for (int i = 0; i < 100; i++) {
		Eigen::JacobiSVD<MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
		MatrixXd U = svd.matrixU(), s = svd.singularValues(), V = svd.matrixV();
		
		MatrixXd S_r = MatrixXd::Zero(s.rows(), s.rows());
		for (int j = 0; j < r; j++) {
			S_r(j, j) = s(j);
		}

		MatrixXd X_new = U * S_r * V.transpose();
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (!std::isnan(Y(i, j))) X_new(i, j) = Y(i, j);
			}
		}

		if ((X - X_new).norm() < tol) break;
		X = X_new;
	}
	return X;
}

int main() {
	auto Y1 = load_csv("../../data/it-singular-val-thresh/Y1.csv");
	auto Y2 = load_csv("../../data/it-singular-val-thresh/Y2.csv");
	auto Y3 = load_csv("../../data/it-singular-val-thresh/Y3.csv");
	auto Y_true = load_csv("../../data/it-singular-val-thresh/Y_true.csv");

	std::cout << (Y_true - it_singular_val_thresh(Y1, 2)).norm() << '\n';
	std::cout << (Y_true - it_singular_val_thresh(Y2, 2)).norm() << '\n';
	std::cout << (Y_true - it_singular_val_thresh(Y3, 2)).norm() << '\n';
}