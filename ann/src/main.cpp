#include <common.hpp>
#include <Dense.hpp>
#include <Conv2D.hpp>
#include <Activation.hpp>
#include <ANN.hpp>
#include <CNN.hpp>
#include <DataLoader.hpp>
#include <data.hpp>
#include <util.hpp>

int main() {
	double lr = 1e-3;
	double lam = 1e-3;
	
	auto [X_train, X_test, y_train, y_test] = load_mnist_digits_data();

	CNN model(
		{	
			new Conv2DL2({1, 28, 28}, 3, 8, lam),
			new ConvTanh({8, 26, 26}),
			new MaxPooling({8, 26, 26}, 2),
		}, {
			new DenseL2(8 * 13 * 13, 64, lam),
			new ReLU(),
			new DenseL2(64, 10, lam),
		},
		new SigmoidBinaryCrossentropy()
	);
	
	for (int epoch = 0; epoch < 1000; epoch++) {
		for (int i = 0; i < X_train.rows(); i++) {
			VectorXd row = X_train.row(i);
			model.predict(Eigen::TensorMap<Tensor3d>(row.data(), 1, 28, 28));
			model.update(y_train.row(i), lr);
		}

		if (epoch < 10 || (epoch + 1) % 10 == 0) {
			std::cout << "[epoch " << epoch + 1 << "] ";
			std::cout << "error rate: " << 100.0 * test_multi_classifier(model, X_test, y_test) << "%\n";
		}
	}
}