#include <Dense.hpp>
#include <data.hpp>

Dense::Dense(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
	weights = MatrixXd::Random(output_size, input_size);
	bias = VectorXd::Random(output_size);
}

Dense::Dense(const std::string& weights_path, const std::string& bias_path) {
	weights = csv2matrix(weights_path);
	bias = csv2vector(bias_path);
}

VectorXd Dense::forward(const VectorXd& input) {
	this->input = input;
	return weights * input + bias;
}

VectorXd Dense::backward(const VectorXd& output_grad, double lr) {
	VectorXd input_grad = weights.transpose() * output_grad;
	weights -= lr * output_grad * input.transpose();
	bias -= lr * output_grad;
	return input_grad;
}

void Dense::save(const std::string& weights_path, const std::string& bias_path) {
	std::ofstream weights_file(weights_path);
	if (weights_file.is_open()) {
		weights_file << weights.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",\n", "", "", "", ""));
		weights_file.close();
	} else {
		throw std::runtime_error("Unable to open file");
	}
	std::ofstream bias_file(bias_path);
	if (bias_file.is_open()) {
		bias_file << bias.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",\n", "", "", "", ""));
		bias_file.close();
	} else {
		throw std::runtime_error("Unable to open file");
	}
}

DenseL2::DenseL2(int input_size, int output_size, double lam) : input_size(input_size), output_size(output_size), lam(lam) {
	weights = MatrixXd::Random(output_size, input_size);
	bias = VectorXd::Random(output_size);
}

DenseL2::DenseL2(const std::string& weights_path, const std::string& bias_path, double lam) : lam(lam) {
	weights = csv2matrix(weights_path);
	bias = csv2vector(bias_path);
}

VectorXd DenseL2::forward(const VectorXd& input) {
	this->input = input;
	return weights * input + bias;
}

VectorXd DenseL2::backward(const VectorXd& output_grad, double lr) {
	VectorXd input_grad = weights.transpose() * output_grad;
	weights -= lr * (output_grad * input.transpose() + lam * weights);
	bias -= lr * (output_grad + lam * bias);
	return input_grad;
}

void DenseL2::save(const std::string& weights_path, const std::string& bias_path) {
	std::ofstream weights_file(weights_path);
	if (weights_file.is_open()) {
		weights_file << weights.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",\n", "", "", "", ""));
		weights_file.close();
	} else {
		throw std::runtime_error("Unable to open file");
	}
	std::ofstream bias_file(bias_path);
	if (bias_file.is_open()) {
		bias_file << bias.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",\n", "", "", "", ""));
		bias_file.close();
	} else {
		throw std::runtime_error("Unable to open file");
	}
}