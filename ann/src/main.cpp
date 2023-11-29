#include <common.hpp>
#include <Dense.hpp>
#include <Activation.hpp>
#include <ANN.hpp>
#include <CNN.hpp>
#include <util.hpp>

int main() {
	auto [X_train, X_test, Y_train, Y_test] = load_mnist_fashion_data();
	int m = X_train.rows(), n_x = X_train.cols(), n_y = Y_train.cols();

	CNN model({
			new Conv2D({1, 28, 28}, 3, 4),
		}, {
			new Dense(4 * 26 * 26, 60),
			new ReLU(),
			new Dense(60, 20),
			new ReLU(),
			new Dense(20, n_y),
		}, 
		new SoftmaxCategoricalCrossentropy()
	);

	for (int epoch = 0; epoch < 500; epoch++) {
		for (int i = 0; i < m; i++) {
			VectorXd row = X_train.row(i);
			Tensor3d t = Eigen::TensorMap<Tensor3d>(row.data(), 1, 28, 28);
			model.predict(t);
			model.update(Y_train.row(i), 0.001);
		}
		if (epoch < 10 || (epoch + 1) % 10 == 0) {
			std::cout << "[epoch " << epoch + 1 << "]";
			std::cout << " train error rate test: " << 100 * test_multi_classifier(model, X_train, Y_train) << "%,";
			std::cout << " test error rate test: " << 100 * test_multi_classifier(model, X_test, Y_test) << "%\n";
		}
	}
}

// ANN: error rate: 15%