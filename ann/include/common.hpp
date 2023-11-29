#ifndef common_hpp
#define common_hpp

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Tensor2d = Eigen::Tensor<double, 2>;
using Tensor3d = Eigen::Tensor<double, 3>;
using Tensor4d = Eigen::Tensor<double, 4>;
using ActivationFn = std::function<VectorXd(const VectorXd&)>;
using LossFn = std::function<VectorXd(const VectorXd&, const VectorXd&)>;

#endif