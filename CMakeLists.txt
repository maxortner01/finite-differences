cmake_minimum_required(VERSION 3.8)

project(poisson)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/extern/eigen)

add_executable(poisson main.cpp)