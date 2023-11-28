#ifndef dense_hpp
#define dense_hpp

#include <common.hpp>
#include <Layer.hpp>

class Dense : public Layer {
public:
	Dense(int input_size, int output_size);
	VectorXd forward(const VectorXd& input) override;
	VectorXd backward(const VectorXd& output_grad, double lr) override;

private:
	int input_size;
	int output_size;
	MatrixXd weights;
	VectorXd bias;
	VectorXd input;
};

#endif