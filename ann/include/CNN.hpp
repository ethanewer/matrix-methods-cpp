#ifndef cnn_hpp
#define cnn_hpp

#include <common.hpp>
#include <Conv2D.hpp>
#include <Layer.hpp>

class CNN {
public:
	CNN(const std::vector<Conv2D*>& conv_layers, const std::vector<Layer*>& dense_layers, LossLayer* loss_layer);
	VectorXd predict(const std::vector<MatrixXd>& input);
	void update(const VectorXd& y_true, double lr);
	~CNN();

private:
	std::vector<Conv2D*> conv_layers;
	std::vector<Layer*> dense_layers;
	LossLayer* loss_layer;
};

#endif