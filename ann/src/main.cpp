#include <common.hpp>
#include <Dense.hpp>
#include <Activation.hpp>
#include <Conv2D.hpp>
#include <ANN.hpp>
#include <CNN.hpp>
#include <DataLoader.hpp>
#include <data.hpp>
#include <util.hpp>

int main() {
	int batch_size = 5000;
	double lr = 0.001;
	
	TensorDataLoader train_data(
		"../../data/cats-vs-dogs/X_train.csv", 
		"../../data/cats-vs-dogs/y_train.csv",
		{1, 100, 100}, 1, batch_size
	);

	TensorDataLoader test_data(
		"../../data/cats-vs-dogs/X_test.csv", 
		"../../data/cats-vs-dogs/y_test.csv",
		{1, 100, 100}, 1, batch_size
	);

	CNN model(
		{
			new Conv2DL2({1, 100, 100}, 3, 8, 1e-3),
			new ConvReLU({8, 98, 98}),
			new MaxPooling({8, 98, 98}, 2),
		}, {
			new DenseL2(8 * 49 * 49, 64, 1e-3),
			new ReLU(),
			new DenseL2(64, 32, 1e-3),
			new ReLU(),
			new DenseL2(32, 1, 1e-3),
		},
		new SigmoidBinaryCrossentropy()
	);

	for (int batch_num = 0; batch_num < 1000; batch_num++) {
		auto [X, Y] = train_data.get_batch();
		for (int epoch = 0; epoch < 5; epoch++) {
			for (int i = 0; i < batch_size; i++) {
				model.predict(X.chip(i, 0));
				model.update(Y.row(i), lr);
			}
		}

		std::cout << "[batch " << batch_num + 1 << "] ";
		std::cout << "error rate: " << 100.0 * test_binary_classifier(model, test_data) << "%\n";
	}
}