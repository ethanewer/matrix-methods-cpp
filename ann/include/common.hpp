#ifndef common_hpp
#define common_hpp

#include <eigen3/Eigen/Dense>
#include <array>
#include <vector>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using ActivationFn = std::function<VectorXd(const VectorXd&)>;
using LossFn = std::function<VectorXd(const VectorXd&, const VectorXd&)>;

#endif