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

VectorXd sigmoid(const VectorXd& v) {
	int n = v.size();
	VectorXd res(n);
	for (int i = 0; i < n; i++) {
		res(i) = 1.0 / (1 + exp(-v(i)));
	}
	return res;
}

VectorXd sigmoid_prime(const VectorXd& v) {
	int n = v.size();
	VectorXd res(n);
	for (int i = 0; i < n; i++) {
		double exp_val = exp(-v(i));
		res(i) = exp_val / pow(1 + exp_val, 2);
	}
	return res;
}

VectorXd mse_prime(const VectorXd& x, const VectorXd& y) {
	return (1.0 / x.size()) * (x - y);
}

VectorXd softmax(const VectorXd& v) {
	VectorXd res = v.array().exp();
	return (1.0 / res.array().sum()) * res;
}

VectorXd softmax_prime(const VectorXd& v) {
	int n = v.size();
	VectorXd s = softmax(v);
	VectorXd res = VectorXd::Zero(n);
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < n; k++) {
			res(i) += s(i) * (static_cast<int>(i == k) - s(k));
		}
	}
	return res;
}

#endif