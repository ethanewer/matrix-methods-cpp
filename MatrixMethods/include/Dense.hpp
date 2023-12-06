#ifndef dense_hpp
#define dense_hpp

#include <common.hpp>
#include <Layer.hpp>

namespace mm {

	class Dense : public Layer {
	public:
		Dense(int input_size, int output_size);
		Dense(const std::string& weights_path, const std::string& bias_path);
		VectorXd forward(const VectorXd& input) override;
		VectorXd backward(const VectorXd& output_grad, double lr) override;
		void save(const std::string& weights_path, const std::string& bias_path);

	private:
		int input_size;
		int output_size;
		MatrixXd weights;
		VectorXd bias;
		VectorXd input;
	};

	class DenseL2 : public Layer {
	public:
		DenseL2(int input_size, int output_size, double lam);
		DenseL2(const std::string& weights_path, const std::string& bias_path, double lam);
		VectorXd forward(const VectorXd& input) override;
		VectorXd backward(const VectorXd& output_grad, double lr) override;
		void save(const std::string& weights_path, const std::string& bias_path);

	private:
		int input_size;
		int output_size;
		double lam;
		MatrixXd weights;
		VectorXd bias;
		VectorXd input;
	};

}

#endif