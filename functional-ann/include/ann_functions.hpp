#ifndef ann_functions_hpp
#define ann_functions_hpp

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd identity(const VectorXd& v) {
	return v;
}

VectorXd identity_prime(const VectorXd& v) {
	return VectorXd::Ones(v.size());
}

VectorXd relu(const VectorXd& v) {
	return v.array().max(0);
}

VectorXd relu_prime(const VectorXd& v) {
	return v.array().max(0).sign();
}

VectorXd clipped_relu(const VectorXd& v) {
	return v.array().max(0.0).min(8.0);
}

VectorXd clipped_relu_prime(const VectorXd& v) {
	int n = v.size();
	VectorXd res(n);
	for (int i = 0; i < n; i++) {
		res(i) = v(i) > 0 && v(i) < 8;
	}
	return res;
}

VectorXd sigmoid(const VectorXd& v) {
	return 1.0 / ((-v.array()).exp() + 1.0);
}

VectorXd sigmoid_prime(const VectorXd& v) {
	VectorXd s = sigmoid(v);
	return s.array() - (s.array() * s.array());
}

VectorXd mse_prime(const VectorXd& x, const VectorXd& y) {
	return (2.0 / x.size()) * (x - y);
}

VectorXd softmax(const VectorXd& v) {
	VectorXd v_exp = (v.array() - v.maxCoeff()).exp();
	return v_exp / v_exp.array().sum();
}

VectorXd softmax_categorical_cross_entropy_prime(const VectorXd& x, const VectorXd& y) {
	return x - y;
}


#endif