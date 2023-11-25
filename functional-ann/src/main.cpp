#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <functional_ann.hpp>
#include <ann_functions.hpp>
#include <util.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
	auto [X_train, X_valid, y_train, y_valid] = load_gd_data();
	int m = X_train.rows(), n = X_train.cols();

	FunctionalANN sigmoid_ann(
		{n, 10, 10, 10, 1},
		{sigmoid, sigmoid, sigmoid, identity},
		{sigmoid_prime, sigmoid_prime, sigmoid_prime, identity_prime},
		mse_prime
	);

	FunctionalANN relu_ann(
		{n, 10, 10, 10, 1},
		{relu, relu, relu, identity},
		{relu_prime, relu_prime, relu_prime, identity_prime},
		mse_prime
	);
	
	for (int epoch = 0; epoch < 500; epoch++) {
		for (int i = 0; i < m; i++) {
			sigmoid_ann.forward(X_train.row(i));
			sigmoid_ann.back_prop_update(y_train.row(i), 0.001);

			relu_ann.forward(X_train.row(i));
			relu_ann.back_prop_update(y_train.row(i), 0.001);
		}

		if ((epoch + 1) % 10 == 0) {
			std::cout << "[epoch " << epoch + 1 << "]";
			std::cout << " sigmoid ann error rate: " << 100 * test_binary_classifier(sigmoid_ann, X_valid, y_valid) << "%,";
			std::cout << " relu ann error rate: " << 100 * test_binary_classifier(relu_ann, X_valid, y_valid) << "%\n";
		}
	}
}
