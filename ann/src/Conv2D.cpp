#include <Conv2D.hpp>

static Tensor2d valid_correlation(const Tensor2d& X, const Tensor2d& K) {
	Tensor2d res = X.convolve(K, Eigen::array<ptrdiff_t, 2> {0, 1});
	return res;
}

static Tensor2d full_colvolution(const Tensor2d& X, const Tensor2d& K) {
	int m_x = X.dimensions()[0], n_x = X.dimensions()[1];
	int m_k = K.dimensions()[0], n_k = K.dimensions()[1];
	Tensor2d X_padded(m_x + 2 * m_k - 2, n_x + 2 * n_k - 2);
	Tensor2d K_rotated = K.reverse(Eigen::array<bool, 2> {true, true});
	for (int i = 0; i < m_x; i++) {
		for (int j = 0; j < n_x; j++) {
			X_padded(i + m_k - 1, j + n_k - 1) = X(i, j);
		}
	}
	return valid_correlation(X_padded, K_rotated);
}

Conv2D::Conv2D(const std::array<int, 3>& input_shape, int kernel_size, int depth) : depth(depth) {
	this->input_shape = input_shape;
	auto [input_depth, input_height, input_width] = input_shape;
	output_shape = {depth, input_height - kernel_size + 1, input_width - kernel_size + 1};
	kernels_shape = {depth, input_depth, kernel_size, kernel_size};
	kernels = (1.0 / (kernel_size * kernel_size)) * Tensor4d(kernels_shape[0], kernels_shape[1], kernels_shape[2], kernels_shape[3]).setRandom();
	biases = Tensor3d(output_shape[0], output_shape[1], output_shape[2]).setRandom();
}

Tensor3d Conv2D::forward(const Tensor3d& input) {
	this->input = input;
	this->output = biases;
	#pragma omp parallel for
	for (int i = 0; i < kernels_shape[0]; i++) {
		for (int j = 0; j < kernels_shape[1]; j++) {
			output.chip(i, 0) += valid_correlation(input.chip(j, 0), kernels.chip(i, 0).chip(j, 0));
		}
	}
	return output;
}

Tensor3d Conv2D::backward(const Tensor3d& output_grad, double lr) {
	Tensor3d input_grad = Tensor3d(input_shape[0], input_shape[1], input_shape[2]).setZero();
	#pragma omp parallel for
	for (int i = 0; i < kernels_shape[0]; i++) {
		for (int j = 0; j < kernels_shape[1]; j++) {
			Tensor2d kernel_grad = valid_correlation(input.chip(j, 0), output_grad.chip(i, 0));
			input_grad.chip(j, 0) += full_colvolution(output_grad.chip(i, 0), kernels.chip(i, 0).chip(j, 0));
			kernels.chip(i, 0).chip(j, 0) -= lr * kernel_grad;
		}
		biases.chip(i, 0) -= lr * output_grad.chip(i, 0);
	}
	return input_grad;
}

Conv2DL2::Conv2DL2(const std::array<int, 3>& input_shape, int kernel_size, int depth, double lam) :  depth(depth), lam(lam) {
	this->input_shape = input_shape;
	auto [input_depth, input_height, input_width] = input_shape;
	output_shape = {depth, input_height - kernel_size + 1, input_width - kernel_size + 1};
	kernels_shape = {depth, input_depth, kernel_size, kernel_size};
	kernels = (1.0 / (kernel_size * kernel_size)) * Tensor4d(kernels_shape[0], kernels_shape[1], kernels_shape[2], kernels_shape[3]).setRandom();
	biases = Tensor3d(output_shape[0], output_shape[1], output_shape[2]).setRandom();
}

Tensor3d Conv2DL2::forward(const Tensor3d& input) {
	this->input = input;
	this->output = biases;
	#pragma omp parallel for
	for (int i = 0; i < kernels_shape[0]; i++) {
		for (int j = 0; j < kernels_shape[1]; j++) {
			output.chip(i, 0) += valid_correlation(input.chip(j, 0), kernels.chip(i, 0).chip(j, 0));
		}
	}
	return output;
}

Tensor3d Conv2DL2::backward(const Tensor3d& output_grad, double lr) {
	Tensor3d input_grad = Tensor3d(input_shape[0], input_shape[1], input_shape[2]).setZero();
	#pragma omp parallel for
	for (int i = 0; i < kernels_shape[0]; i++) {
		for (int j = 0; j < kernels_shape[1]; j++) {
			Tensor2d kernel_grad = valid_correlation(input.chip(j, 0), output_grad.chip(i, 0)) + lam * kernels.chip(i, 0).chip(j, 0);
			input_grad.chip(j, 0) += full_colvolution(output_grad.chip(i, 0), kernels.chip(i, 0).chip(j, 0));
			kernels.chip(i, 0).chip(j, 0) -= lr * kernel_grad;
		}
		biases.chip(i, 0) -= lr * (output_grad.chip(i, 0) + lam * biases.chip(i, 0));
	}
	return input_grad;
}

MaxPooling::MaxPooling(const std::array<int, 3>& input_shape, int pool_size) : pool_size(pool_size) {
	this->input_shape = input_shape;
	output_shape = {input_shape[0], input_shape[1] / pool_size, input_shape[2] / pool_size};
}

Tensor3d MaxPooling::forward(const Tensor3d& input) {
	input_grad = Tensor3d(input_shape[0], input_shape[1], input_shape[2]).setZero();
	Tensor3d output(output_shape[0], output_shape[1], output_shape[2]);

	#pragma omp parallel for
	for (int i = 0; i < output_shape[0]; i++) {
		for (int j = 0; j < output_shape[1]; j++) {
			for (int k = 0; k < output_shape[2]; k++) {
				int max_x = pool_size * k, max_y = pool_size * k;
				for (int x = pool_size * j; x < pool_size * (j + 1); x++) {
					for (int y = pool_size * k; y < pool_size * (k + 1); y++) {
						if (input(i, x, y) > input(i, max_x, max_y)) {
							max_x = x;
							max_y = y;
						}
					}
				}
				output(i, j, k) = input(i, max_x, max_y);
				input_grad(i, max_x, max_y) = 1;
			}
		}
	}

	return output;
}

Tensor3d MaxPooling::backward(const Tensor3d& output_grad, double lr) {
	#pragma omp parallel for
	for (int i = 0; i < output_shape[0]; i++) {
		for (int j = 0; j < output_shape[1]; j++) {
			for (int k = 0; k < output_shape[2]; k++) {
				for (int x = pool_size * j; x < pool_size * (j + 1); x++) {
					for (int y = pool_size * k; y < pool_size * (k + 1); y++) {
						input_grad(i, x, y) *= output_grad(i, j, k);
					}
				}
			}
		}
	}
	return input_grad;
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

StandardScaler::StandardScaler(const std::array<int, 3>& input_shape) {
	this->input_shape = input_shape;
	this->output_shape = input_shape;
}

Tensor3d StandardScaler::forward(const Tensor3d& input) {
	// Tensor3d t = input;
	// VectorXd v = Eigen::Map<VectorXd>(t.data(), t.size());
	// std::cout << v.hasNaN() << '\n';

	double mean = std::reduce(input.data(), input.data() + input.size()) / input.size();
	Tensor3d tmp = (input - mean).square();
	scale_factor = 1.0 / sqrt(std::reduce(tmp.data(), tmp.data() + tmp.size()));

	return scale_factor * (input - mean);
}

Tensor3d StandardScaler::backward(const Tensor3d& output_grad, double lr) {
	return scale_factor * output_grad;
}