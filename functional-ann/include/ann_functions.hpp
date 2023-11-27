#ifndef ann_functions_hpp
#define ann_functions_hpp

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd identity(const VectorXd& z) {
	return z;
}

VectorXd relu(const VectorXd& z) {
	return z.array().max(0);
}

VectorXd relu_prime(const VectorXd& a) {
	return a.array().max(0).sign();
}

VectorXd sigmoid(const VectorXd& v) {
	return 1.0 / ((-v.array()).exp() + 1.0);
}

VectorXd sigmoid_prime(const VectorXd& a) {
	return a.array() * (1.0 - a.array());
}

VectorXd sigmoid_binary_cross_entropy_prime(const VectorXd& a, const VectorXd& y) {
	return a.array() * (1.0 - y.array()) - y.array() * (1.0 - a.array());
}

VectorXd mse_prime(const VectorXd& a, const VectorXd& y) {
	return (2.0 / a.size()) * (a - y);
}

VectorXd softmax(const VectorXd& v) {
	VectorXd v_exp = (v.array() - v.maxCoeff()).exp();
	return v_exp / v_exp.array().sum();
}

VectorXd softmax_categorical_cross_entropy_prime(const VectorXd& a, const VectorXd& y) {
	return a - y;
}

#endif