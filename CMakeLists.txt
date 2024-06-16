cmake_minimum_required(VERSION 3.8)

project(poisson)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-w -O3)

add_executable(poisson main.cpp)

target_include_directories(poisson
    PRIVATE
        ${CMAKE_SOURCE_DIR}/extern/eigen
)