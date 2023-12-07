#include <MatrixMethods>
#include <load_data.hpp>
#include <util.hpp>

int main() {
	int batch_size = 5000;
	double lr = 1e-3;

	mm::MatrixDataLoader train_data(
		"../../data/mnist-fashion/X_train.csv", 
		"../../data/mnist-fashion/y_train.csv",
		784, 10, batch_size
	);

	mm::MatrixDataLoader test_data(
		"../../data/mnist-fashion/X_test.csv", 
		"../../data/mnist-fashion/y_test.csv",
		784, 10, batch_size
	);

	std::string model_path = "../models/fashion/ann-1/model_";

	mm::ANN model(
		{
			new mm::DenseL2(model_path + "layer_0_weights.csv", model_path + "layer_0_bias.csv", 1e-3),
			new mm::ReLU(),
			new mm::DenseL2(model_path + "layer_2_weights.csv", model_path + "layer_2_bias.csv", 1e-3),
			new mm::ReLU(),
			new mm::DenseL2(model_path + "layer_4_weights.csv", model_path + "layer_4_bias.csv", 1e-3),
		},
		new mm::SoftmaxCategoricalCrossentropy()
	);

	for (int batch_num = 0; batch_num < 1000; batch_num++) {
		auto [X, Y] = train_data.get_batch();
		for (int epoch = 0; epoch < 10; epoch++) {
			for (int i = 0; i < batch_size; i++) {
				model.predict(X.row(i));
				model.update(Y.row(i), lr);
			}
		}
		if (batch_num < 10 || (batch_num + 1) % 10 == 0) {
			std::cout << "[batch " << batch_num + 1 << "] ";
			std::cout << "error rate: " << 100.0 * test_multi_classifier(model, test_data) << "%\n";
		}
	}
}