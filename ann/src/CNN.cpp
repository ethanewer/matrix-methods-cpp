#include <CNN.hpp>
#include <Conv2D.hpp>
#include <Dense.hpp>

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

void CNN::save(const std::string& prefix) {
	for (int i = 0; i < conv_layers.size(); i++) {
		if (Conv2D* conv = dynamic_cast<Conv2D*>(conv_layers[i])) {
			std::string file_prefix = prefix + "_conv_" + std::to_string(i);
			conv->save(file_prefix  + "_kernels.csv", file_prefix + "_biases.csv");
		} else if (Conv2DL2* conv_l2 = dynamic_cast<Conv2DL2*>(conv_layers[i])) {
			std::string file_prefix = prefix + "_conv_" + std::to_string(i);
			conv_l2->save(file_prefix  + "_kernels.csv", file_prefix + "_biases.csv");
		}
	}

	for (int i = 0; i < dense_layers.size(); i++) {
		if (Dense* dense = dynamic_cast<Dense*>(dense_layers[i])) {
			std::string file_prefix = prefix + "_dense_" + std::to_string(i);
			dense->save(file_prefix  + "_weights.csv", file_prefix + "_bias.csv");
		} else if (DenseL2* dense_l2 = dynamic_cast<DenseL2*>(dense_layers[i])) {
			std::string file_prefix = prefix + "_dense_" + std::to_string(i);
			dense_l2->save(file_prefix  + "_weights.csv", file_prefix + "_bias.csv");
		}
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