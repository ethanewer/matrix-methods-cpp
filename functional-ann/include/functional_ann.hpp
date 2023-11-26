#ifndef functional_ann_hpp
#define functional_ann_hpp

#include <eigen3/Eigen/Dense>
#include <vector>
#include <functional>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using ActivationFn = std::function<VectorXd(const VectorXd&)>;
using LossFn = std::function<VectorXd(const VectorXd&, const VectorXd&)>;

struct FunctionalANN {
	FunctionalANN(
		const std::vector<int>& layer_sizes, 
		const std::vector<ActivationFn>& a_fns,
		const std::vector<ActivationFn>& a_fn_primes,
		const LossFn& loss_fn_prime
	) : layer_sizes(layer_sizes), num_layers(layer_sizes.size()) {
		if (a_fns.size() != num_layers - 1) {
			throw std::runtime_error("a_fns.size() != layer_sizes.size() - 1");
		} else if (a_fn_primes.size() != num_layers - 2) {
			throw std::runtime_error("a_fn_primes.size() != layer_sizes.size() - 2");
		}
		this->a_fns = a_fns;
		this->a_fn_primes = a_fn_primes;
		this->loss_fn_prime = loss_fn_prime;

		for (int i = 0; i < num_layers; i++) {
			if (i > 0) {
				W.push_back(MatrixXd::Random(layer_sizes[i], layer_sizes[i - 1]));
			}
			a.push_back(VectorXd::Zero(layer_sizes[i]));
		}
	}

	void forward(const VectorXd& input) {
		a[0] = input;
		for (int i = 0; i < num_layers - 1; i++) {
			a[i + 1] = a_fns[i](W[i] * a[i]);
			if (a[i + 1].hasNaN()) {
				std::cout << (a[i]).transpose() << '\n';
				std::cout << (W[i] * a[i]).transpose() << '\n';
				std::cout << a[i + 1].transpose() << '\n';
				throw std::runtime_error("[forward] NaN");
			}
		}
	}

	void back_prop_update(const VectorXd& y, double learning_rate) {	
		VectorXd z_prime = loss_fn_prime(a.back(), y);
		if (z_prime.hasNaN()) {
			std::cout << a.back().transpose() << '\n';
			std::cout << loss_fn_prime(a.back(), y).transpose() << '\n';
			throw std::runtime_error("[back_prop_update (z_prime before loop)] NaN");
		}
		for (int i = num_layers - 2; i >= 0; i--) {
			MatrixXd W_prime = z_prime * a[i].transpose();
			if (i > 0) {
				z_prime = (W[i].transpose() * z_prime).array() * a_fn_primes[i - 1](a[i]).array();
			}
			if (z_prime.hasNaN()) throw std::runtime_error("[back_prop_update (z_prime)] NaN");
			if (W_prime.hasNaN()) throw std::runtime_error("[back_prop_update (W_prime)] NaN");

			W[i] -= learning_rate * W_prime;
		}
	}

	VectorXd predict(const VectorXd& input) {
		forward(input);
		return a.back();
	}

	int num_layers;
	std::vector<int> layer_sizes;
	std::vector<ActivationFn> a_fns;
	std::vector<ActivationFn> a_fn_primes;
	LossFn loss_fn_prime;
	std::vector<MatrixXd> W;
	std::vector<VectorXd> a;
};

#endif