#ifndef ann_hpp
#define ann_hpp

#include <common.hpp>
#include <Layer.hpp>

class ANN {
public:
	ANN(const std::vector<Layer*>& layers, LossLayer* loss_layer);
	VectorXd predict(const VectorXd& input);
	void update(const VectorXd& y_true, double lr);
	void save(const std::string& file_prefix);
	~ANN();

private:
	std::vector<Layer*> layers;
	LossLayer* loss_layer;
};

#endif