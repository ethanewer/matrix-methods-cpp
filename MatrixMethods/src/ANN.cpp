#include <ANN.hpp>
#include <Dense.hpp>
using namespace mm;

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

void ANN::save(const std::string& prefix) {
	for (int i = 0; i < layers.size(); i++) {
		if (Dense* dense = dynamic_cast<Dense*>(layers[i])) {
			std::string file_prefix = prefix + "_layer_" + std::to_string(i);
			dense->save(file_prefix  + "_weights.csv", file_prefix + "_bias.csv");
		} else if (DenseL2* dense_l2 = dynamic_cast<DenseL2*>(layers[i])) {
			std::string file_prefix = prefix + "_layer_" + std::to_string(i);
			dense_l2->save(file_prefix  + "_weights.csv", file_prefix + "_bias.csv");
		}
	}
}


ANN::~ANN() {
	for (Layer* layer : layers) {
		delete layer;
	}
	delete loss_layer;
}