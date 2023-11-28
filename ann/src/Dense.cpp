#include <Dense.hpp>

Dense::Dense(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
	weights = MatrixXd::Random(output_size, input_size);
	bias = VectorXd::Random(output_size);
}

VectorXd Dense::forward(const VectorXd& input) {
	this->input = input;
	return weights * input + bias;
}

VectorXd Dense::backward(const VectorXd& output_grad, double lr) {
	VectorXd input_grad = weights.transpose() * output_grad;
	weights -= lr * output_grad * input.transpose();
	bias -= lr * output_grad;
	return input_grad;
}