#include <Conv2D.hpp>
#include <data.hpp>

using namespace mm;

Conv2D::Conv2D(const std::array<int, 3>& input_shape, int kernel_size, int depth) : depth(depth) {
	this->input_shape = input_shape;
	auto [input_depth, input_height, input_width] = input_shape;
	output_shape = {depth, input_height - kernel_size + 1, input_width - kernel_size + 1};
	kernels_shape = {depth, input_depth, kernel_size, kernel_size};
	kernels = (1.0 / (kernel_size * kernel_size)) * Tensor4d(kernels_shape[0], kernels_shape[1], kernels_shape[2], kernels_shape[3]).setRandom();
	biases = Tensor3d(output_shape[0], output_shape[1], output_shape[2]).setRandom();
}

Conv2D::Conv2D(
	const std::array<int, 3>& input_shape, 
	int kernel_size, int depth, 
	const std::string& kernels_path, 
	const std::string& biases_path
) : depth(depth) {
	this->input_shape = input_shape;
	auto [input_depth, input_height, input_width] = input_shape;
	output_shape = {depth, input_height - kernel_size + 1, input_width - kernel_size + 1};
	kernels_shape = {depth, input_depth, kernel_size, kernel_size};
	
	VectorXd kernels_data = csv2vector(kernels_path);
	kernels = Eigen::TensorMap<Tensor4d>(kernels_data.data(), kernels_shape[0], kernels_shape[1], kernels_shape[2], kernels_shape[3]);
	
	VectorXd biases_data = csv2vector(biases_path);
	biases = Eigen::TensorMap<Tensor3d>(biases_data.data(), output_shape[0], output_shape[1], output_shape[2]);
}

Tensor3d Conv2D::forward(const Tensor3d& input) {
	this->input = input;
	output = biases;
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
			input_grad.chip(j, 0) += full_convolution(output_grad.chip(i, 0), kernels.chip(i, 0).chip(j, 0));
			kernels.chip(i, 0).chip(j, 0) -= lr * kernel_grad;
		}
		biases.chip(i, 0) -= lr * output_grad.chip(i, 0);
	}
	return input_grad;
}

void Conv2D::save(const std::string& kernels_path, const std::string& biases_path) {
	std::ofstream kernels_file(kernels_path);
	if (kernels_file.is_open()) {
		VectorXd data = Eigen::Map<VectorXd>(kernels.data(), kernels.size());
		kernels_file << data.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",\n", "", "", "", ""));
		kernels_file.close();
	} else {
		throw std::runtime_error("Unable to open file");
	}
	std::ofstream biases_file(biases_path);
	if (biases_file.is_open()) {
		VectorXd data = Eigen::Map<VectorXd>(biases.data(), biases.size());
		biases_file << data.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",\n", "", "", "", ""));
		biases_file.close();
	} else {
		throw std::runtime_error("Unable to open file");
	}
}

Conv2DL2::Conv2DL2(const std::array<int, 3>& input_shape, int kernel_size, int depth, double lam) :  depth(depth), lam(lam) {
	this->input_shape = input_shape;
	auto [input_depth, input_height, input_width] = input_shape;
	output_shape = {depth, input_height - kernel_size + 1, input_width - kernel_size + 1};
	kernels_shape = {depth, input_depth, kernel_size, kernel_size};
	kernels = (1.0 / (kernel_size * kernel_size)) * Tensor4d(kernels_shape[0], kernels_shape[1], kernels_shape[2], kernels_shape[3]).setRandom();
	biases = Tensor3d(output_shape[0], output_shape[1], output_shape[2]).setRandom();
}

Conv2DL2::Conv2DL2(
	const std::array<int, 3>& input_shape, 
	int kernel_size, int depth, double lam,
	const std::string& kernels_path, 
	const std::string& biases_path
) : depth(depth), lam(lam) {
	this->input_shape = input_shape;
	auto [input_depth, input_height, input_width] = input_shape;
	output_shape = {depth, input_height - kernel_size + 1, input_width - kernel_size + 1};
	kernels_shape = {depth, input_depth, kernel_size, kernel_size};
	
	VectorXd kernels_data = csv2vector(kernels_path);
	kernels = Eigen::TensorMap<Tensor4d>(kernels_data.data(), kernels_shape[0], kernels_shape[1], kernels_shape[2], kernels_shape[3]);
	
	VectorXd biases_data = csv2vector(biases_path);
	biases = Eigen::TensorMap<Tensor3d>(biases_data.data(), output_shape[0], output_shape[1], output_shape[2]);
}

Tensor3d Conv2DL2::forward(const Tensor3d& input) {
	this->input = input;
	output = biases;
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
			input_grad.chip(j, 0) += full_convolution(output_grad.chip(i, 0), kernels.chip(i, 0).chip(j, 0));
			kernels.chip(i, 0).chip(j, 0) -= lr * kernel_grad;
		}
		biases.chip(i, 0) -= lr * (output_grad.chip(i, 0) + lam * biases.chip(i, 0));
	}
	return input_grad;
}

void Conv2DL2::save(const std::string& kernels_path, const std::string& biases_path) {
	std::ofstream kernels_file(kernels_path);
	if (kernels_file.is_open()) {
		VectorXd data = Eigen::Map<VectorXd>(kernels.data(), kernels.size());
		kernels_file << data.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",\n", "", "", "", ""));
		kernels_file.close();
	} else {
		throw std::runtime_error("Unable to open file");
	}
	std::ofstream biases_file(biases_path);
	if (biases_file.is_open()) {
		VectorXd data = Eigen::Map<VectorXd>(biases.data(), biases.size());
		biases_file << data.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",\n", "", "", "", ""));
		biases_file.close();
	} else {
		throw std::runtime_error("Unable to open file");
	}
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

static Tensor2d mm::valid_correlation(const Tensor2d& X, const Tensor2d& K) {
	return X.convolve(K, Eigen::array<ptrdiff_t, 2> {0, 1});
}

static Tensor2d mm::full_convolution(const Tensor2d& X, const Tensor2d& K) {
	int m_x = X.dimensions()[0], n_x = X.dimensions()[1];
	int m_k = K.dimensions()[0], n_k = K.dimensions()[1];
	Tensor2d X_padded(m_x + 2 * m_k - 2, n_x + 2 * n_k - 2);
	Tensor2d K_rotated = K.reverse(Eigen::array<bool, 2> {true, true});
	for (int i = 0; i < m_x; i++) {
		for (int j = 0; j < n_x; j++) {
			X_padded(i + m_k - 1, j + n_k - 1) = X(i, j);
		}
	}
	return X_padded.convolve(K_rotated, Eigen::array<ptrdiff_t, 2> {0, 1});
}