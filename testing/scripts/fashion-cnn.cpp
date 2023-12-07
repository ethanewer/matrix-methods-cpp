#include <MatrixMethods>
#include <load_data.hpp>
#include <util.hpp>

int main() {
	double lr = 5e-4;
	double lam = 1e-3;
	
	auto [X_train, X_test, y_train, y_test] = load_mnist_fashion_data();

	std::string model_path = "../models/fashion/cnn-1/model";

	// CNN model(
	// 	{	
	// 		new Conv2DL2({1, 28, 28}, 3, 8, lam),
	// 		new ConvTanh({8, 26, 26}),
	// 		new MaxPooling({8, 26, 26}, 2),
	// 	}, {
	// 		new DenseL2(8 * 13 * 13, 64, lam),
	// 		new ReLU(),
	// 		new DenseL2(64, 10, lam),
	// 	},
	// 	new SigmoidBinaryCrossentropy()
	// );
	
	mm::CNN model(
		{	
			new mm::Conv2DL2({1, 28, 28}, 3, 8, lam,  model_path + "_conv_0_kernels.csv", model_path + "_conv_0_biases.csv"),
			new mm::ConvTanh({8, 26, 26}),
			new mm::MaxPooling({8, 26, 26}, 2),
		}, {
			new mm::DenseL2(model_path + "_dense_0_weights.csv", model_path + "_dense_0_bias.csv", lam),
			new mm::ReLU(),
			new mm::DenseL2(model_path + "_dense_2_weights.csv", model_path + "_dense_2_bias.csv", lam),
		},
		new mm::SigmoidBinaryCrossentropy()
	);

	double min_error_rate = test_multi_classifier(model, X_test, y_test);
	
	for (int epoch = 0; epoch < 1000; epoch++) {
		for (int i = 0; i < X_train.rows(); i++) {
			VectorXd row = X_train.row(i);
			model.predict(Eigen::TensorMap<Tensor3d>(row.data(), 1, 28, 28));
			model.update(y_train.row(i), lr);
		}

		double error_rate = test_multi_classifier(model, X_test, y_test);
		if (error_rate < min_error_rate) {
			min_error_rate = error_rate;
			model.save(model_path);
		}
		if (epoch < 10 || (epoch + 1) % 10 == 0) {
			std::cout << "[epoch " << epoch + 1 << "] ";
			std::cout << "lowest error rate: " << 100.0 * min_error_rate << "%\n";
		}
	}
}