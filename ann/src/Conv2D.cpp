#include <Conv2D.hpp>

Tensor3d scale_down(const Tensor3d& t) {
	double abs_sum = std::reduce(t.data(), t.data() + t.size(), 0.0, [](double a, double b) {
		return a + std::abs(b);
	});
	double scale_factor = abs_sum > t.size() ? static_cast<double>(t.size()) / abs_sum : 1.0;
	return scale_factor * t;
}

Tensor2d valid_correlation(const Tensor2d& X, const Tensor2d& K) {
	Tensor2d res = X.convolve(K, Eigen::array<ptrdiff_t, 2> {0, 1});
	return res;
}

Tensor2d full_colvolution(const Tensor2d& X, const Tensor2d& K) {
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

Conv2D::Conv2D(std::array<int, 3> input_shape, int kernel_size, int depth) : input_shape(input_shape), depth(depth) {
	auto [input_depth, input_height, input_width] = input_shape;
	output_shape = {depth, input_height - kernel_size + 1, input_width - kernel_size + 1};
	kernels_shape = {depth, input_depth, kernel_size, kernel_size};
	kernels = Tensor4d(kernels_shape[0], kernels_shape[1], kernels_shape[2], kernels_shape[3]).setRandom();
	biases = Tensor3d(output_shape[0], output_shape[1], output_shape[2]).setRandom();
}

Tensor3d Conv2D::forward(const Tensor3d& input) {
	this->input = input;
	this->output = biases;
	for (int i = 0; i < kernels_shape[0]; i++) {
		for (int j = 0; j < kernels_shape[1]; j++) {
			output.chip(i, 0) += valid_correlation(input.chip(j, 0), kernels.chip(i, 0).chip(j, 0));
		}
	}
	return scale_down(output);
}

Tensor3d Conv2D::backward(const Tensor3d& output_grad, double lr) {
	Tensor3d input_grad = Tensor3d(input_shape[0], input_shape[1], input_shape[2]).setZero();
	for (int i = 0; i < kernels_shape[0]; i++) {
		for (int j = 0; j < kernels_shape[1]; j++) {
			Tensor2d kernel_grad = valid_correlation(input.chip(j, 0), output_grad.chip(i, 0));
			input_grad.chip(j, 0) += full_colvolution(output_grad.chip(i, 0), kernels.chip(i, 0).chip(j, 0));
			kernels.chip(i, 0).chip(j, 0) -= lr * kernel_grad;
		}
		biases.chip(i, 0) -= lr * output_grad.chip(i, 0);
	}
	return scale_down(input_grad);
}