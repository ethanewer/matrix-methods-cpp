#ifndef conv2d_hpp
#define conv2d_hpp

#include <common.hpp>
#include <Layer.hpp>

class Conv2D : public ConvLayer {
public:
	Conv2D(const std::array<int, 3>& input_shape, int kernel_size, int depth);
	Tensor3d forward(const Tensor3d& input) override;
	Tensor3d backward(const Tensor3d& output_grad, double lr) override;

	int depth;
	std::array<int, 4> kernels_shape;
	Tensor4d kernels;
	Tensor3d biases;
	Tensor3d input;
	Tensor3d output;
};

class Conv2DL2 : public ConvLayer {
public:
	Conv2DL2(const std::array<int, 3>& input_shape, int kernel_size, int depth, double lam);
	Tensor3d forward(const Tensor3d& input) override;
	Tensor3d backward(const Tensor3d& output_grad, double lr) override;

	int depth;
	std::array<int, 4> kernels_shape;
	Tensor4d kernels;
	Tensor3d biases;
	Tensor3d input;
	Tensor3d output;
	double lam;
};

#endif