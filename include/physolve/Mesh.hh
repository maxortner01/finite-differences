#pragma once

#include <algorithm>
#include <Eigen/Eigen>
#include <Eigen/Cholesky>

namespace ps
{
    template<typename T, uint32_t N_>
    struct Mesh
    {
        constexpr static uint32_t N = N_;
        
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> space;

        Mesh() :
            space(N, N)
        { space.setZero(); }
    };

    // Generates a mesh of equivalent dimension as the input that contains the value
    // of the Laplacian of the input field at each point
    template<typename T, uint32_t N>
    auto
    laplacian(const Mesh<T, N>& mesh)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ret(N, N);
        for (int32_t y = 0; y < N; y++)
            for (int32_t x = 0; x < N; x++)
            {
                auto value = mesh.space(y, x) * -1.0;
                value += 0.25 * mesh.space(y, fmax(x - 2, fmax(x - 1, 0)));
                value += 0.25 * mesh.space(fmax(y - 2, fmax(y - 1, 0)), x);
                value += 0.25 * mesh.space(y, fmin(x + 2, fmin(x + 1, N - 1)));
                value += 0.25 * mesh.space(fmin(y + 2, fmin(y + 1, N - 1)), x);
                ret(y, x) = value;
            }
        return ret;
    }
}
