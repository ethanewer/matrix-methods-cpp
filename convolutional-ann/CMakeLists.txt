cmake_minimum_required(VERSION 3.14)
project(Main)

find_package(Eigen3 REQUIRED)
include_directories(/opt/homebrew/include)

add_executable(main src/main.cpp)

target_include_directories(main PRIVATE include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set_property(TARGET main PROPERTY CXX_STANDARD 20)
