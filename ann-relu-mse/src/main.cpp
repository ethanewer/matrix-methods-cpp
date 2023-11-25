#include <eigen3/Eigen/Dense>
#include <iostream>
#include <ann.hpp>
#include <util.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
	auto [X_train, X_valid, y_train, y_valid] = load_credit_card_fraud_data();
	int m = X_train.rows(), n = X_train.cols();

	ANN ann({n, 10, 10, 10, 1});
	
	for (int epoch = 0; epoch < 500; epoch++) {
		for (int i = 0; i < m; i++) {
			ann.forward(X_train.row(i));
			ann.back_prop_update(y_train.row(i), 0.002);
		}

		if ((epoch + 1) % 10 == 0) {
			std::cout << "[epoch " << epoch + 1 << "]"; 
			std::cout << " error rate " << 100 * test_binary_classifier(ann, X_valid, y_valid) << "%\n";
		}
	}
}
