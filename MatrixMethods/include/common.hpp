#ifndef common_hpp
#define common_hpp

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <omp.h>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <utility>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Tensor2d = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using Tensor3d = Eigen::Tensor<double, 3, Eigen::RowMajor>;
using Tensor4d = Eigen::Tensor<double, 4, Eigen::RowMajor>;

#endif