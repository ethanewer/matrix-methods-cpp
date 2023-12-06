#ifndef conv2d_hpp
#define conv2d_hpp

#include <common.hpp>
#include <Layer.hpp>

namespace mm {

	class Conv2D : public ConvLayer {
	public:
		Conv2D(const std::array<int, 3>& input_shape, int kernel_size, int depth);
		Conv2D(const std::array<int, 3>& input_shape, int kernel_size, int depth, const std::string& kernels_path, const std::string& biases_path);
		Tensor3d forward(const Tensor3d& input) override;
		Tensor3d backward(const Tensor3d& output_grad, double lr) override;
		void save(const std::string& kernels_path, const std::string& biases_path);

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
		Conv2DL2(const std::array<int, 3>& input_shape, int kernel_size, int depth, double lam, const std::string& kernels_path, const std::string& biases_path);
		Tensor3d forward(const Tensor3d& input) override;
		Tensor3d backward(const Tensor3d& output_grad, double lr) override;
		void save(const std::string& kernels_path, const std::string& biases_path);

		int depth;
		std::array<int, 4> kernels_shape;
		Tensor4d kernels;
		Tensor3d biases;
		Tensor3d input;
		Tensor3d output;
		double lam;
	};

	class MaxPooling : public ConvLayer {
	public:
		MaxPooling(const std::array<int, 3>& input_shape, int pool_size);
		Tensor3d forward(const Tensor3d& input) override;
		Tensor3d backward(const Tensor3d& output_grad, double lr) override;

		int pool_size;
		Tensor3d input_grad;
	};

	static Tensor2d valid_correlation(const Tensor2d& X, const Tensor2d& K);

	static Tensor2d full_convolution(const Tensor2d& X, const Tensor2d& K);

}

#endif