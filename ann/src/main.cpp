#include <common.hpp>
#include <Dense.hpp>
#include <Activation.hpp>
#include <ANN.hpp>
#include <CNN.hpp>
#include <data.hpp>
#include <util.hpp>

int main() {
	auto [X_train, X_test, Y_train, Y_test] = load_mnist_fashion_data();
	int m = X_train.rows(), n_x = X_train.cols(), n_y = Y_train.cols();

	// ANN model(
	// 	{
	// 		new DenseL2(n_x, 96, 1e-3),
	// 		new ReLU(),
	// 		new DenseL2(96, 32, 1e-3),
	// 		new ReLU(),
	// 		new DenseL2(32, n_y, 1e-3),
	// 	},
	// 	new SoftmaxCategoricalCrossentropy()
	// );

	std::string model_path = "../models/784-96-32-10/model";

	ANN model(
		{
			new DenseL2(model_path + "_layer_0_weights.csv", model_path + "_layer_0_bias.csv", 1e-3),
			new ReLU(),
			new DenseL2(model_path + "_layer_2_weights.csv", model_path + "_layer_2_bias.csv", 1e-3),
			new ReLU(),
			new DenseL2(model_path + "_layer_4_weights.csv", model_path + "_layer_4_bias.csv", 1e-3),
		},
		new SoftmaxCategoricalCrossentropy()
	);

	double min_error_rate = 1;
	for (int epoch = 0; epoch < 1000; epoch++) {
		for (int i = 0; i < m; i++) {
			model.predict(X_train.row(i));
			model.update(Y_train.row(i), 1e-4);
		}
		if (epoch < 10 || (epoch + 1) % 10 == 0) {
			double error_rate = test_multi_classifier(model, X_test, Y_test);
			if (error_rate < min_error_rate) {
				min_error_rate = error_rate;
				model.save("../models/784-96-32-10/model");
			}

			std::cout << "[epoch " << epoch + 1 << "]";
			std::cout << " test error rate: " << 100 * error_rate << "%";
			std::cout << std::endl;
		}
	}
}

// best error rate: 11.24%
// ANN model(
// 	{
// 		new DenseL2(28 * 28, 96, 1e-3),
// 		new ReLU(),
// 		new DenseL2(96, 32, 1e-3),
// 		new ReLU(),
// 		new DenseL2(32, n_y, 1e-3),
// 	},
// 	new SoftmaxCategoricalCrossentropy()
// );