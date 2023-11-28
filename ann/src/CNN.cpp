#include <CNN.hpp>

CNN::CNN(
	const std::vector<Conv2D*>& conv_layers,
	const std::vector<Layer*>& dense_layers,
	LossLayer* loss_layer
) : conv_layers(conv_layers), dense_layers(dense_layers), loss_layer(loss_layer) {}

VectorXd CNN::predict(const std::vector<MatrixXd>& input) {
	std::vector<MatrixXd> output_tensor = input;
	for (Conv2D* layer : conv_layers) {
		output_tensor = layer->forward(output_tensor);
	}

	int flattened_size = output_tensor.size() * output_tensor[0].size();
	VectorXd output_vector(flattened_size);
	int vec_idx = 0;
	for (int i = 0; i < output_tensor.size(); i++) {
		for (int j = 0; j < output_tensor[0].rows(); j++) {
			for (int k = 0; k < output_tensor[0].cols(); k++) {
				output_vector(vec_idx++) = output_tensor[i](j, k);
			}
		}
	}
	for (Layer* layer : dense_layers) {
		output_vector = layer->forward(output_vector);
	}
	return loss_layer->forward(output_vector);
}

void CNN::update(const VectorXd& y_true, double lr) {
	VectorXd grad = loss_layer->backward(y_true, lr);
	for (int i = dense_layers.size() - 1; i >= 0; i--) {
		grad = dense_layers[i]->backward(grad, lr);
	}

	auto [d, m, n] = conv_layers.back()->output_shape;

	std::vector<MatrixXd> conv_grad = std::vector<MatrixXd>(d, MatrixXd(m, n));
	int vec_idx = 0;
	for (int i = 0; i < d; i++) {
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < n; k++) {
				conv_grad[i](j, k) = grad(vec_idx++);
			}
		}
	}

	for (int i = conv_layers.size() - 1; i >= 0; i--) {
		conv_grad = conv_layers[i]->backward(conv_grad, lr);
	}
}

CNN::~CNN() {
	for (Conv2D* layer : conv_layers) {
		delete layer;
	}
	for (Layer* layer : dense_layers) {
		delete layer;
	}
	delete loss_layer;
}