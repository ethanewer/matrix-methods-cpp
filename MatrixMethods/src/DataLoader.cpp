#include <DataLoader.hpp>

using namespace mm;

MatrixDataLoader::MatrixDataLoader(
	const std::string& data_path, 
	const std::string& label_path, 
	int data_row_size,
	int label_row_size,
	int batch_size
) : 
	data_path(data_path), 
	label_path(label_path), 
	data_row_size(data_row_size),
	label_row_size(label_row_size),
	batch_size(batch_size),
	data_file(std::ifstream(data_path)), 
	label_file(std::ifstream(label_path)) {}

std::tuple<MatrixXd, MatrixXd> MatrixDataLoader::get_batch() {
	MatrixXd X(batch_size, data_row_size);
	MatrixXd Y(batch_size, label_row_size);

	std::string data_row, label_row, cell;
	for (int i = 0; i < batch_size; i++) {
		if (!std::getline(data_file, data_row) || !std::getline(label_file, label_row)) {
			data_file.close();
			label_file.close();
			data_file.open(data_path);
			label_file.open(label_path);
		}

		std::istringstream data_ss(data_row);
		for (int j = 0; std::getline(data_ss, cell, ','); j++) {
			X(i, j) = std::stod(cell);
		}

		std::istringstream label_ss(label_row);
		for (int j = 0; std::getline(label_ss, cell, ','); j++) {
			Y(i, j) = std::stod(cell);
		}
	}

	return {X, Y};
}

TensorDataLoader::TensorDataLoader(
	const std::string& data_path, 
	const std::string& label_path, 
	const std::array<int, 3>& data_shape, 
	int label_row_size, 
	int batch_size
) : 
	data_path(data_path), 
	label_path(label_path), 
	d(data_shape[0]),
	m(data_shape[1]),
	n(data_shape[2]),
	label_row_size(label_row_size),
	batch_size(batch_size),
	data_file(std::ifstream(data_path)), 
	label_file(std::ifstream(label_path)) {}

std::tuple<Tensor4d, MatrixXd> TensorDataLoader::get_batch() {
	Tensor4d X(batch_size, d, m, n);
	MatrixXd Y(batch_size, label_row_size);

	std::string data_row, label_row, cell;
	for (int i = 0; i < batch_size; i++) {
		if (!std::getline(data_file, data_row) || !std::getline(label_file, label_row)) {
			data_file.close();
			label_file.close();
			data_file.open(data_path);
			label_file.open(label_path);
		}

		std::istringstream data_ss(data_row);
		for (int j = 0; std::getline(data_ss, cell, ','); j++) {
			X(i, j / (m * n), (j % (m * n)) / n, j % n) = std::stod(cell);
		}

		std::istringstream label_ss(label_row);
		for (int j = 0; std::getline(label_ss, cell, ','); j++) {
			Y(i, j) = std::stod(cell);
		}
	}

	return {X, Y};
}
