cmake_minimum_required(VERSION 3.14)
project(MatrixMethods)

find_package(Eigen3 REQUIRED)
include_directories(/opt/homebrew/include)

file(GLOB SOURCES "src/*.cpp")

# Create a library instead of an executable
add_library(matrix_methods_library STATIC ${SOURCES})

# Set compile options for the library
target_compile_options(matrix_methods_library PRIVATE -Xpreprocessor -fopenmp)

# Set include directories for the library
target_include_directories(matrix_methods_library PUBLIC include)
target_include_directories(matrix_methods_library PRIVATE /opt/homebrew/Cellar/libomp/17.0.6/include)

# Set link options for the library
target_link_options(matrix_methods_library PRIVATE -L/opt/homebrew/Cellar/libomp/17.0.6/lib -lomp)

# Set compiler flags for the library
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
 
# Set C++ standard for the library
set_property(TARGET matrix_methods_library PROPERTY CXX_STANDARD 20)
