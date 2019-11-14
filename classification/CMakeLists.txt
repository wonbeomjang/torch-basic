cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(vgg)

find_package(Torch REQUIRED)

add_executable(vgg vgg.cpp)
target_link_libraries(vgg "${TORCH_LIBRARIES}")
set_property(TARGET vgg PROPERTY CXX_STANDARD 11)