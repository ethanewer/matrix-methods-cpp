#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <functional_ann.hpp>
#include <ann_functions.hpp>
#include <util.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
	auto [X_train, X_test, Y_train, Y_test] = load_credit_card_fraud_data();
	int m = X_train.rows(), n_x = X_train.cols(), n_y = Y_train.cols();

	FunctionalANN model(
		{n_x, 20, 20, 20, n_y},
		{sigmoid, sigmoid, sigmoid, sigmoid},
		{sigmoid_prime, sigmoid_prime, sigmoid_prime},
		sigmoid_binary_cross_entropy_prime
	);
	
	for (int epoch = 0; epoch < 200; epoch++) {
		for (int i = 0; i < m; i++) {
			model.forward(X_train.row(i));
			model.back_prop_update(Y_train.row(i), 0.005);
		}
		if ((epoch + 1) % 10 == 0) {
			std::cout << "[epoch " << epoch + 1 << "]";
			std::cout << " error rate: " << 100 * test_binary_classifier(model, X_test, Y_test) << "%\n";
		}
	}
}