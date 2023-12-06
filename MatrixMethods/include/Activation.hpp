#ifndef activaton_hpp
#define activaton_hpp

#include <common.hpp>
#include <Layer.hpp>

namespace mm {

	class ReLU : public Layer {
	public:
		VectorXd forward(const VectorXd& input) override;
		VectorXd backward(const VectorXd& output_grad, double lr) override;

	private:
		VectorXd activation;
	};

	class ClippedReLU : public Layer {
	public:
		VectorXd forward(const VectorXd& input) override;
		VectorXd backward(const VectorXd& output_grad, double lr) override;

	private:
		VectorXd input;
	};


	class Sigmoid : public Layer {
	public:
		VectorXd forward(const VectorXd& input) override;
		VectorXd backward(const VectorXd& output_grad, double lr) override;

	private:
		VectorXd activation;
	};

	class SigmoidBinaryCrossentropy : public LossLayer {
	public:
		VectorXd forward(const VectorXd& input);
		VectorXd backward(const VectorXd& y_true, double lr);

	private:
		VectorXd activation;
	};

	class SoftmaxCategoricalCrossentropy : public LossLayer {
	public:
		VectorXd forward(const VectorXd& input);
		VectorXd backward(const VectorXd& y_true, double lr);

	private:
		VectorXd activation;
	};

	class ConvReLU : public ConvLayer {
	public:
		ConvReLU(const std::array<int, 3>& input_shape);
		Tensor3d forward(const Tensor3d& input) override;
		Tensor3d backward(const Tensor3d& output_grad, double lr) override;

		Tensor3d activation;
	};

	class ConvSigmoid : public ConvLayer {
	public:
		ConvSigmoid(const std::array<int, 3>& input_shape);
		Tensor3d forward(const Tensor3d& input) override;
		Tensor3d backward(const Tensor3d& output_grad, double lr) override;

		Tensor3d activation;
	};

	class ConvTanh : public ConvLayer {
	public:
		ConvTanh(const std::array<int, 3>& input_shape);
		Tensor3d forward(const Tensor3d& input) override;
		Tensor3d backward(const Tensor3d& output_grad, double lr) override;

		Tensor3d activation;
	};

}

#endif