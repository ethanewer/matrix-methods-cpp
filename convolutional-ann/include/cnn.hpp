#ifndef cnn_hpp
#define cnn_hpp

#include <eigen3/Eigen/Dense>
#include <vector>
#include <array>
#include <utility>
#include <functional>
#include <iostream>

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using ActivationFn = std::function<VectorXd(const VectorXd&)>;
using LossFn = std::function<VectorXd(const VectorXd&, const VectorXd&)>;

struct CNN {
	CNN(
		int input_size,
		const vector<int>& kernal_sizes, 
		const vector<int>& kernal_depths, 
		const vector<int>& dense_layer_sizes, 
		const vector<ActivationFn>& a_fns,
		const vector<ActivationFn>& a_fn_primes,
		const LossFn& loss_fn_prime
	) : 
		input_size(input_size),
		kernal_sizes(kernal_sizes),
		kernal_depths(kernal_depths),
		dense_layer_sizes(dense_layer_sizes), 
		num_dense_layers(dense_layer_sizes.size() + 1) 
	{
		if (kernal_sizes.size() != kernal_depths.size()) {
			throw std::runtime_error("kernal_sizes.size() != kernal_depths.size()");
		}
		num_conv_layers = kernal_sizes.size() + 1;
		conv_layer_sizes.push_back({1, input_size, input_size});
		a_conv.push_back(vector<MatrixXd>(1, MatrixXd::Zero(input_size, input_size)));
		for (int i = 0; i < num_conv_layers - 1; i++) {
			conv_layer_sizes.push_back({
				kernal_depths[i], 
				conv_layer_sizes[i][1] - kernal_sizes[i] + 1, 
				conv_layer_sizes[i][1] - kernal_sizes[i] + 1
			});
			a_conv.push_back(vector<MatrixXd>(
				kernal_depths[i], 
				MatrixXd::Zero(
					conv_layer_sizes[i][1] - kernal_sizes[i] + 1, 
					conv_layer_sizes[i][1] - kernal_sizes[i] + 1
				)
			));
			
			int K_size = conv_layer_sizes[i][0] * conv_layer_sizes[i + 1][0] * kernal_sizes[i] * kernal_sizes[i];
			K.push_back(vector<vector<MatrixXd>>(
				conv_layer_sizes[i][0], vector<MatrixXd>(
					conv_layer_sizes[i + 1][0], MatrixXd::Random(kernal_sizes[i], kernal_sizes[i])
				)	
			));
			b_conv.push_back(vector<MatrixXd>(
				kernal_depths[i], 
				MatrixXd::Random(
					conv_layer_sizes[i][1] - kernal_sizes[i] + 1, 
					conv_layer_sizes[i][1] - kernal_sizes[i] + 1
				)
			));
		}
		
		if (a_fns.size() != num_dense_layers - 1) {
			throw std::runtime_error("a_fns.size() != dense_layer_sizes.size()");
		} else if (a_fn_primes.size() != num_dense_layers - 2) {
			throw std::runtime_error("a_fn_primes.size() != dense_layer_sizes.size() - 1");
		}
		this->a_fns = a_fns;
		this->a_fn_primes = a_fn_primes;
		this->loss_fn_prime = loss_fn_prime;

		a_dense.push_back(VectorXd::Zero(a_conv.back().size() * a_conv.back().back().rows() * a_conv.back().back().cols()));
		for (int i = 0; i < num_dense_layers - 1; i++) {
			W.push_back(MatrixXd::Random(dense_layer_sizes[i], a_dense.back().size()));
			b_dense.push_back(VectorXd::Random(dense_layer_sizes[i]));
			a_dense.push_back(VectorXd::Zero(dense_layer_sizes[i]));
		}
	}

	void forward(const VectorXd& input) {
		a_conv[0][0] = input.reshaped(input_size, input_size);
		for (int i = 0; i < num_conv_layers - 1; i++) {
			for (int j = 0; j < conv_layer_sizes[i][0]; j++) {
				a_conv[i + 1][j] = b_conv[i][j];
				for (int k = 0; k < conv_layer_sizes[i + 1][0]; k++) {
					a_conv[i + 1][k] += valid_correlation(a_conv[i][j], K[i][j][k]);
				}
			}
		}

		int s = a_conv.back().back().rows() * a_conv.back().back().cols();
		for (int i = 0; i < a_conv.back().size(); i++) {
			a_dense[0].segment(i * s, s) = a_conv.back()[i].reshaped(s, 1);
		}

		for (int i = 0; i < num_dense_layers - 1; i++) {
			a_dense[i + 1] = a_fns[i](W[i] * a_dense[i] + b_dense[i]);
		}
	}

	void back_prop_update(const VectorXd& y, double learning_rate) {	
		VectorXd z_dense_prime = loss_fn_prime(a_dense.back(), y);
		for (int i = num_dense_layers - 2; i > 0; i--) {
			VectorXd next_z_prime = (W[i].transpose() * z_dense_prime).array() * a_fn_primes[i - 1](a_dense[i]).array();
			W[i] -= learning_rate * z_dense_prime * a_dense[i].transpose();
			b_dense[i] -= learning_rate * z_dense_prime;
			z_dense_prime = std::move(next_z_prime);
		}
		W[0] -= learning_rate * z_dense_prime * a_dense[0].transpose();
		b_dense[0] -= learning_rate * z_dense_prime;
		z_dense_prime = (W[0].transpose() * z_dense_prime).array() * a_dense[0].array();

		int s = a_conv.back().back().rows() * a_conv.back().back().cols();
		vector<MatrixXd> z_conv_prime;
		for (int i = 0; i < a_conv.back().size(); i++) {
			z_conv_prime.push_back(
				z_dense_prime.segment(i * s, s).reshaped(a_conv.back().back().rows(), a_conv.back().back().cols())
			);
		}
		
		for (int i = num_conv_layers - 2; i > 0; i--) {
			vector<MatrixXd> next_z_conv_prime(
				conv_layer_sizes[i - 1][0], 
				MatrixXd::Zero(conv_layer_sizes[i - 1][1], conv_layer_sizes[i - 1][2])
			);
			for (int j = 0; j < conv_layer_sizes[i - 1][0]; j++) {
				for (int k = 0; k < conv_layer_sizes[i][0]; k++) {
					K[i][j][k] -= learning_rate * valid_correlation(a_conv[i][j], z_conv_prime[k]);
					next_z_conv_prime[j] += full_colvolution(a_conv[i][j], K[i][j][k]);
				}
				b_conv[i][j] -= learning_rate * z_conv_prime[j];
			}
			z_conv_prime = std::move(next_z_conv_prime);
		}
	}

	MatrixXd valid_correlation(const MatrixXd& X, const MatrixXd& K) {
		int m_x = X.rows(), n_x = X.cols();
		int m_k = K.rows(), n_k = K.cols();
		int m = m_x - m_k + 1, n = n_x - n_k + 1;
		MatrixXd res(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				res(i, j) = (X.block(i, j, m_k, n_k).array() * K.array()).sum();
			}
		}
		return res;
	}

	MatrixXd full_colvolution(const MatrixXd& X, const MatrixXd& K) {
		int m_x = X.rows(), n_x = X.cols();
		int m_k = K.rows(), n_k = K.cols();
		MatrixXd X_padded(m_x + 2 * m_k - 2, n_x + 2 * n_k - 2);
		MatrixXd K_rotated = K.reverse();
		for (int i = 0; i < m_x; i++) {
			for (int j = 0; j < n_x; j++) {
				X_padded(i + m_k - 1, j + n_k - 1) = X(i, j);
			}
		}
		return valid_correlation(X_padded, K_rotated);
	}


	VectorXd predict(const VectorXd& input) {
		forward(input);
		return a_dense.back();
	}

	int input_size;
	int num_conv_layers;
	vector<std::array<int, 3>> conv_layer_sizes;
	vector<int> kernal_sizes;
	vector<int> kernal_depths;
	int num_dense_layers;
	vector<int> dense_layer_sizes;
	vector<ActivationFn> a_fns;
	vector<ActivationFn> a_fn_primes;
	LossFn loss_fn_prime;
	vector<vector<vector<MatrixXd>>> K;
	vector<vector<MatrixXd>> b_conv;
	vector<vector<MatrixXd>> a_conv;
	vector<MatrixXd> W;
	vector<VectorXd> b_dense;
	vector<VectorXd> a_dense;
};

#endif