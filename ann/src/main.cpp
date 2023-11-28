#include <common.hpp>
#include <Dense.hpp>
#include <Activation.hpp>
#include <ANN.hpp>
#include <CNN.hpp>
#include <util.hpp>

int main() {
	auto [X_train, X_test, Y_train, Y_test] = load_mnist_digits_data();
	int m = X_train.rows(), n_x = X_train.cols(), n_y = Y_train.cols();

	CNN model(
		{
			new Conv2D({1, 28, 28}, 3, 1),
			new Conv2D({1, 26, 26}, 3, 1),
		},
		{
			new Dense(1 * 24 * 24, 20),
			new ReLU(),
			new Dense(20, n_y),
		}, 
		new SoftmaxCategoricalCrossentropy()
	);

	for (int epoch = 0; epoch < 250; epoch++) {
		for (int i = 0; i < m; i++) {
			model.predict({X_train.row(i).reshaped(28, 28)});
			model.update(Y_train.row(i), 0.001);
		}
		if ((epoch + 1) % 5 == 0) {
			std::cout << "[epoch " << epoch + 1 << "]";
			// std::cout << " error rate train: " << 100 * test_multi_classifier(model, X_train, Y_train) << "%,";
			std::cout << " error rate test: " << 100 * test_multi_classifier(model, X_test, Y_test) << "%\n";
		}
	}
}
