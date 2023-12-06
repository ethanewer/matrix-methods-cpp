#ifndef layer_hpp
#define layer_hpp

#include <common.hpp>

namespace mm {

	class Layer {
	public:
		virtual VectorXd forward(const VectorXd& input) = 0;
		virtual VectorXd backward(const VectorXd& output_grad, double lr) = 0;
		virtual ~Layer() = default;
	};

	class LossLayer {
	public:
		virtual VectorXd forward(const VectorXd& input) = 0;
		virtual VectorXd backward(const VectorXd& output_grad, double lr) = 0;
		virtual ~LossLayer() = default;
	};

	class ConvLayer {
	public:
		virtual Tensor3d forward(const Tensor3d& input) = 0;
		virtual Tensor3d backward(const Tensor3d& output_grad, double lr) = 0;
		virtual ~ConvLayer() = default;

		std::array<int, 3> input_shape;
		std::array<int, 3> output_shape;
	};

}

#endif