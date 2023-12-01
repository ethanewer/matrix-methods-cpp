#ifndef conv2d_hpp
#define conv2d_hpp

#include <common.hpp>
#include <Layer.hpp>

class Conv2D {
public:
	Conv2D(const std::array<int, 3>& input_shape, int kernel_size, int depth);
	virtual Tensor3d forward(const Tensor3d& input);
	virtual Tensor3d backward(const Tensor3d& output_grad, double lr);

	int depth;
	std::array<int, 3> input_shape;
	std::array<int, 3> output_shape;
	std::array<int, 4> kernels_shape;
	Tensor4d kernels;
	Tensor3d biases;
	Tensor3d input;
	Tensor3d output;
};

class Conv2DL2 : public Conv2D {
public:
	Conv2DL2(const std::array<int, 3>& input_shape, int kernel_size, int depth, double lam);
	Tensor3d forward(const Tensor3d& input) override;
	Tensor3d backward(const Tensor3d& output_grad, double lr) override;

	double lam;
};

#endif