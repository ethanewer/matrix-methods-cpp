#include <Activation.hpp>

VectorXd ReLU::forward(const VectorXd& input) {
	return this->activation = input.array().max(0);
}

VectorXd ReLU::backward(const VectorXd& output_grad, double lr) {
	return output_grad.array() * activation.array().sign();
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
	VectorXd exp_input = input.array().exp();
	return this->activation = exp_input / exp_input.sum();
}

VectorXd SoftmaxCategoricalCrossentropy::backward(const VectorXd& y_true, double lr) {
	return activation - y_true;
}