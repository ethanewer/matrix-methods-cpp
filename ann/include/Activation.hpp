#ifndef activaton_hpp
#define activaton_hpp

#include <common.hpp>
#include <Layer.hpp>

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

#endif