#include <Activation.hpp>

using namespace mm;

VectorXd ReLU::forward(const VectorXd& input) {
	return this->activation = input.array().max(0);
}

VectorXd ReLU::backward(const VectorXd& output_grad, double lr) {
	return output_grad.array() * activation.array().sign();
}

VectorXd ClippedReLU::forward(const VectorXd& input) {
	this->input = input;
	return input.array().max(0).min(4);
}

VectorXd ClippedReLU::backward(const VectorXd& output_grad, double lr) {
	int n = input.size();
	VectorXd grad = output_grad;
	for (int i = 0; i < n; i++) {
		if (input(i) < 0 || 4 < input(i)) grad(i) = 0;
	}
	return grad;
}

VectorXd Sigmoid::forward(const VectorXd& input) {
	return this->activation = 1.0 / ((-input.array()).exp() + 1.0);
}

VectorXd Sigmoid::backward(const VectorXd& output_grad, double lr) {
	return output_grad.array() * activation.array() * (1.0 - activation.array());
}

VectorXd SigmoidBinaryCrossentropy::forward(const VectorXd& input) {
	return this->activation = 1.0 / ((-input.array()).exp() + 1.0);
}

VectorXd SigmoidBinaryCrossentropy::backward(const VectorXd& y_true, double lr) {
	return activation.array() * (1.0 - y_true.array()) - y_true.array() * (1.0 - activation.array());
}

VectorXd SoftmaxCategoricalCrossentropy::forward(const VectorXd& input) {
	VectorXd exp_input = input.array().exp();\
	return this->activation = exp_input / exp_input.sum();
}

VectorXd SoftmaxCategoricalCrossentropy::backward(const VectorXd& y_true, double lr) {
	return activation - y_true;
}

ConvReLU::ConvReLU(const std::array<int, 3>& input_shape) {
	this->input_shape = input_shape;
	this->output_shape = input_shape;
}

Tensor3d ConvReLU::forward(const Tensor3d& input) {
	activation = input.cwiseMax(0.0);
	return activation;
}

Tensor3d ConvReLU::backward(const Tensor3d& output_grad, double lr) {
	return output_grad * activation.sign();
}

ConvSigmoid::ConvSigmoid(const std::array<int, 3>& input_shape) {
	this->input_shape = input_shape;
	this->output_shape = input_shape;
}

Tensor3d ConvSigmoid::forward(const Tensor3d& input) {
	activation = 1.0 / (1.0 + input.exp());
	return activation;
}

Tensor3d ConvSigmoid::backward(const Tensor3d& output_grad, double lr) {
	return output_grad * activation * (1 - activation);
}


ConvTanh::ConvTanh(const std::array<int, 3>& input_shape) {
	this->input_shape = input_shape;
	this->output_shape = input_shape;
}

Tensor3d ConvTanh::forward(const Tensor3d& input) {
	Tensor3d exp_input = input.exp();
	Tensor3d exp_neg_input = (-input).exp();
	activation = (exp_input - exp_neg_input) / (exp_input + exp_neg_input);
	return activation;
}

Tensor3d ConvTanh::backward(const Tensor3d& output_grad, double lr) {
	return output_grad * (1.0 - activation.square());
}