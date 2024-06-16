#pragma once

#include "../../Mesh.hh"

namespace ps::Solvers::Temporal
{
    template<typename T>
    struct Vector2 { T x, y; };

    template<typename _Matrix>
    Vector2<float>
    gradient(const _Matrix& field, int x, int y, int SIZE)
    {
        return Vector2<float>{
            (float)(field(y, fmin(x + 1, SIZE - 1)) - field(y, fmax(x - 1, 0))) * 0.5f,
            (float)(field(fmin(y + 1, SIZE - 1), x) - field(fmax(y - 1, 0), x)) * 0.5f
        };
    }

    // Solves the equations $d^2 \phi/dt^2 = v^2 (\nabla^2\phi - f(x, y))$
    template<typename T, uint32_t N>
    struct Wave
    {
        double t = 0.0;
        const T v;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> first_deriv;

        Wave(T _v, const Mesh<T, N>& mesh) : 
            v(_v), first_deriv(mesh.space.rows(), mesh.space.cols())
        { first_deriv.setZero(); }

        void updateMesh(
            Mesh<T, N>& mesh, 
            const std::function<T(const Eigen::Array<double, 1, 2>&, double t)>& rhs, 
            const double& dt)
        {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> func(mesh.space.rows(), mesh.space.cols());
            for (uint32_t y = 0; y < func.rows(); y++)
                for (uint32_t x = 0; x < func.cols(); x++)
                    func(y, x) = rhs(Eigen::Array<double, 1, 2>({ x, y }), t);

            //decltype(func) diff(mesh.space.rows(), mesh.space.cols());
            decltype(func) diff = v*v * (laplacian(mesh) - func);

            for (uint32_t r = 0; r < diff.rows(); r++)
                for (uint32_t c = 0; c < diff.cols(); c++)
                    if (abs(diff(r, c)) <= 1e-6) diff(r, c) = 0;

            // Temporary copy of the next time-step
            Mesh<T, N> new_mesh;
            new_mesh.space = mesh.space;
            new_mesh.space += diff * dt;

            decltype(func) diff2 = v * v * (laplacian(new_mesh) - func);

            first_deriv += (diff + diff2) * dt / 2.0;

            // Update the boundary
            for (int i = 1; i < mesh.space.cols() - 1; i++)
            {
                first_deriv(0, i) = v * (mesh.space(1, i) - mesh.space(0, i)) * 0.5;
                first_deriv(N - 1, i) = v * (mesh.space(N - 2, i) - mesh.space(N - 1, i)) * 0.5;
                first_deriv(i, 0) = v * (mesh.space(i, 1) - mesh.space(i, 0)) * 0.5;
                first_deriv(i, N - 1) = v * (mesh.space(i, N - 2) - mesh.space(i, N - 1)) * 0.5;
            }

            mesh.space  += first_deriv * dt;
            t += dt;
        }
    };
}
