#ifndef data_hpp
#define data_hpp

#include <common.hpp>

VectorXd csv2vector(const std::string& path);

MatrixXd csv2matrix(const std::string& path);

MatrixXd csv2matrix_with_ones(const std::string& path);

VectorXd csv2vector(const std::string& path, int max_rows);

MatrixXd csv2matrix(const std::string& path, int max_rows);

MatrixXd normalize(const MatrixXd& X);

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> split_data(const MatrixXd& X, const VectorXd& y, double a, double b);

std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> load_credit_card_fraud_data();

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> load_mnist_digits_data();

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> load_mnist_fashion_data();

#endif