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
	int train_batch_size = 1000;
	double lr = 0.0001;

	TensorDataLoader train_data(
		"../../data/cats-vs-dogs/X_train.csv", 
		"../../data/cats-vs-dogs/y_train.csv",
		{1, 100, 100}, 1, train_batch_size
	);

	TensorDataLoader test_data(
		"../../data/cats-vs-dogs/X_test.csv", 
		"../../data/cats-vs-dogs/y_test.csv",
		{1, 100, 100}, 1, 5000
	);

	CNN model(
		{
			new Conv2D({1, 100, 100}, 3, 16),
		}, {
			new Dense(16 * 98 * 98, 128),
			new ReLU(),
			new Dense(128, 1),
		},
		new SigmoidBinaryCrossentropy()
	);

	for (int batch_num = 0; batch_num < 1000; batch_num++) {
		auto [X, Y] = train_data.get_batch();
		for (int epoch = 0; epoch < 5; epoch++) {
			std::cout << "    [epoch " << epoch + 1 << "]\n";
			for (int i = 0; i < train_batch_size; i++) {
				model.predict(X.chip(i, 0));
				model.update(Y.row(i), lr);
			}
		}

		std::cout << "[batch " << batch_num + 1 << "] ";
		std::cout << "error rate: " << 100.0 * test_binary_classifier(model, test_data) << "%\n";
	}
}