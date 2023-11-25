#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <eigen3/Eigen/Dense>

using Eigen::VectorXi;
using Eigen::MatrixXd;

inline double dist(MatrixXd&& a, MatrixXd&& b) {
	return (a - b).norm();
}

std::tuple<MatrixXd, VectorXi> k_means(MatrixXd& X, int k) {
	int m = X.rows(), n = X.cols();
	
	MatrixXd X_T = X.transpose();
	
	std::random_device rd;
  std::mt19937 gen(rd()); 
	std::uniform_int_distribution<int> distribution(0, n - 1);
	MatrixXd centroids;
	centroids.resize(k, m);
	for (int l = 0; l < k; l++) {
		centroids.row(l) = X_T.row(distribution(gen));
	} 
	
	VectorXi C;
	C.resize(n);
	
	for (int i = 0; i < 20; i++) {
		for (int j = 0; j < n; j++) {
			C(j) = 0;
			double min_dist = dist(X_T.row(j), centroids.row(0));
			for (int l = 1; l < k; l++) {
				double d = dist(X_T.row(j), centroids.row(l));
				if (d < min_dist) {
					C(j) = l;
					min_dist = d;
				}
			}
		}
		
		for (int l = 0; l < k; l++) {
			centroids.row(l) = MatrixXd::Zero(1, m);
			int count = 0;
			for (int j = 0; j < n; j++) {
				if (C(j) == l) {
					centroids.row(l) += X_T.row(j);
					count++;
				}
			}
			if (count > 0) centroids.row(l) /= count;
			else centroids.row(l) = X_T.row(distribution(gen));
		}
	}
	return {centroids.transpose(), C};
}

int main() {
	MatrixXd A {
		{3, 3, 3, -1, -1, -1}, 
		{1, 1, 1, -3, -3, -3}, 
		{1, 1, 1, -3, -3, -3}, 
		{3, 3, 3, -1, -1, -1},
	};
	
	auto [centroids, C] = k_means(A, 2);

	std::cout << A << "\n\n" << centroids << "\n\n" << C << '\n';
}