#ifndef cnn_hpp
#define cnn_hpp

#include <common.hpp>
#include <Layer.hpp>

class CNN {
public:
	CNN(const std::vector<ConvLayer*>& conv_layers, const std::vector<Layer*>& dense_layers, LossLayer* loss_layer);
	VectorXd predict(const Tensor3d& input);
	void update(const VectorXd& y_true, double lr);
	~CNN();

private:
	std::vector<ConvLayer*> conv_layers;
	std::vector<Layer*> dense_layers;
	LossLayer* loss_layer;
};

#endif