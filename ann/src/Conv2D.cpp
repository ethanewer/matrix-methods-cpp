#include <Conv2D.hpp>

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

Conv2D::Conv2D(std::array<int, 3> input_shape, int kernel_size, int depth) : input_shape(input_shape), depth(depth) {
	auto [input_depth, input_height, input_width] = input_shape;
	output_shape = {depth, input_height - kernel_size + 1, input_width - kernel_size + 1};
	kernels_shape = {depth, input_depth, kernel_size, kernel_size};
	kernels = std::vector<std::vector<MatrixXd>>(kernels_shape[0], std::vector<MatrixXd>(kernels_shape[1]));
	for (int i = 0; i < kernels_shape[0]; i++) {
		for (int j = 0; j < kernels_shape[1]; j++) {
			kernels[i][j] = MatrixXd::Random(kernels_shape[2], kernels_shape[3]);
		}
	}
	biases = std::vector<MatrixXd>(output_shape[0]);
	for (int i = 0; i < output_shape[0]; i++) {
		biases[i] = MatrixXd::Random(output_shape[1], output_shape[2]);
	}
}

std::vector<MatrixXd> Conv2D::forward(const std::vector<MatrixXd>& input) {
	this->input = input;
	this->output = biases;
	for (int i = 0; i < kernels_shape[0]; i++) {
		for (int j = 0; j < kernels_shape[1]; j++) {
			output[i] += valid_correlation(input[j], kernels[i][j]);
		}
	}
	return output;
}

std::vector<MatrixXd> Conv2D::backward(const std::vector<MatrixXd>& output_grad, double lr) {
	std::vector<MatrixXd> input_grad = std::vector<MatrixXd>(
		input_shape[0], 
		MatrixXd::Zero(input_shape[1], input_shape[2])
	);

	for (int i = 0; i < kernels_shape[0]; i++) {
		for (int j = 0; j < kernels_shape[1]; j++) {
			MatrixXd kernel_grad = valid_correlation(input[j], output_grad[i]);
			input_grad[j] += full_colvolution(output_grad[i], kernels[i][j]);
			kernels[i][j] -= lr * kernel_grad;
		}
		biases[i] -= lr * output_grad[i];
	}
	
	return input_grad;
}