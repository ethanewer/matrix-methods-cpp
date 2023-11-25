#ifndef ann_hpp
#define ann_hpp

#include <eigen3/Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct ANN {
	ANN(const std::vector<int>& layer_sizes) 
		: layer_sizes(layer_sizes), num_layers(layer_sizes.size()) {
		
		for (int i = 0; i < num_layers; i++) {
			if (i > 0) {
				W.push_back(MatrixXd::Random(layer_sizes[i], layer_sizes[i - 1]));
			}
			a.push_back(VectorXd::Zero(layer_sizes[i]));
		}
	}

	void forward(const VectorXd& input) {
		a[0] = input;
		for (int i = 0; i < num_layers - 2; i++) {
			a[i + 1] = (W[i] * a[i]).array().max(0);
		}
		a.back() = W[num_layers - 2] * a[num_layers - 2];
	}

	void back_prop_update(const VectorXd& y, double learning_rate) {	
		VectorXd z_prime = (1.0 / y.size()) * (a.back() - y);
		for (int i = num_layers - 2; i >= 0; i--) {
			MatrixXd W_prime = z_prime * a[i].transpose();
			if (i > 0) {
				z_prime = (W[i].transpose() * z_prime).array() * a[i].array().sign();
			}
			W[i] -= learning_rate * W_prime;
		}
	}

	VectorXd predict(const VectorXd& input) {
		forward(input);
		return a.back();
	}

	int num_layers;
	std::vector<int> layer_sizes;
	std::vector<MatrixXd> W;
	std::vector<VectorXd> a;
};

#endif