#ifndef data_hpp
#define data_hpp

#include <common.hpp>

namespace mm {

    VectorXd csv2vector(const std::string& path);

    MatrixXd csv2matrix(const std::string& path);

    MatrixXd csv2matrix_with_ones(const std::string& path);

    VectorXd csv2vector(const std::string& path, int max_rows);

    MatrixXd csv2matrix(const std::string& path, int max_rows);

    MatrixXd normalize(const MatrixXd& X);

    std::tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> split_data(const MatrixXd& X, const VectorXd& y, double a, double b);

}

#endif