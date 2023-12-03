cmake_minimum_required(VERSION 3.14)
project(Main)

find_package(Eigen3 REQUIRED)
include_directories(/opt/homebrew/include)

file(GLOB SOURCES "src/*.cpp")

add_executable(main ${SOURCES})

target_compile_options(main PRIVATE -Xpreprocessor -fopenmp)

target_include_directories(main PRIVATE include)
target_include_directories(main PRIVATE /opt/homebrew/Cellar/libomp/17.0.6/include)

target_link_options(main PRIVATE -L/opt/homebrew/Cellar/libomp/17.0.6/lib -lomp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set_property(TARGET main PROPERTY CXX_STANDARD 20)
