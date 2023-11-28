#ifndef conv2d_hpp
#define conv2d_hpp

#include <common.hpp>
#include <Layer.hpp>

class Conv2D {
public:
	Conv2D(std::array<int, 3> input_shape, int kernel_size, int depth);
	std::vector<MatrixXd> forward(const std::vector<MatrixXd>& input);
	std::vector<MatrixXd> backward(const std::vector<MatrixXd>& output_grad, double lr);

	int depth;
	std::array<int, 3> input_shape;
	std::array<int, 3> output_shape;
	std::array<int, 4> kernels_shape;
	std::vector<std::vector<MatrixXd>> kernels;
	std::vector<MatrixXd> biases;
	std::vector<MatrixXd> input;
	std::vector<MatrixXd> output;
};

#endif