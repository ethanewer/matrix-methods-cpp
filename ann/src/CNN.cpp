#include <CNN.hpp>

CNN::CNN(
	const std::vector<ConvLayer*>& conv_layers,
	const std::vector<Layer*>& dense_layers,
	LossLayer* loss_layer
) : conv_layers(conv_layers), dense_layers(dense_layers), loss_layer(loss_layer) {}

VectorXd CNN::predict(const Tensor3d& input) {
	Tensor3d output_tensor = input;
	for (ConvLayer* layer : conv_layers) {
		output_tensor = layer->forward(output_tensor);
	}

	VectorXd output_vector = Eigen::Map<VectorXd>(output_tensor.data(), output_tensor.size());

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
	Tensor3d conv_grad = Eigen::TensorMap<Tensor3d>(grad.data(), d, m, n);

	for (int i = conv_layers.size() - 1; i >= 0; i--) {
		conv_grad = conv_layers[i]->backward(conv_grad, lr);
	}
}

CNN::~CNN() {
	for (ConvLayer* layer : conv_layers) {
		delete layer;
	}
	for (Layer* layer : dense_layers) {
		delete layer;
	}
	delete loss_layer;
}