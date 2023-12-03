#ifndef data_loader_hpp
#define data_loader_hpp

#include <common.hpp>
#include <data.hpp>

class MatrixDataLoader {
public:
	MatrixDataLoader(
		const std::string& data_path, 
		const std::string& label_path, 
		int data_size, 
		int label_size, 
		int batch_size
	);
	
	std::tuple<MatrixXd, MatrixXd> get_batch();

private:
	std::string data_path;
	std::string label_path;
	int data_size;
	int label_size;
	int batch_size;
	std::ifstream data_file;
	std::ifstream label_file;
};

class TensorDataLoader {
public:
	TensorDataLoader(
		const std::string& data_path, 
		const std::string& label_path, 
		const std::array<int, 3>& data_shape, 
		int label_size, 
		int batch_size
	);
	
	std::tuple<Tensor4d, MatrixXd> get_batch();

private:
	std::string data_path;
	std::string label_path;
	int d;
	int m;
	int n;
	int label_size;
	int batch_size;
	std::ifstream data_file;
	std::ifstream label_file;
};


#endif
