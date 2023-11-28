#ifndef layer_hpp
#define layer_hpp

#include <common.hpp>

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

#endif