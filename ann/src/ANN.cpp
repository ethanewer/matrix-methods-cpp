#include <ANN.hpp>

ANN::ANN(const std::vector<Layer*>& layers, LossLayer* loss_layer) : layers(layers), loss_layer(loss_layer) {}

VectorXd ANN::predict(const VectorXd& input) {
	VectorXd output = input;
	for (Layer* layer : layers) {
		output = layer->forward(output);
	}
	return loss_layer->forward(output);
}

void ANN::update(const VectorXd& y_true, double lr) {
	VectorXd grad = loss_layer->backward(y_true, lr);
	for (int i = layers.size() - 1; i >= 0; i--) {
		grad = layers[i]->backward(grad, lr);
	}
}

ANN::~ANN() {
	for (Layer* layer : layers) {
		delete layer;
	}
	delete loss_layer;
}