#pragma once

#include "../../Mesh.hh"

namespace ps::Solvers::Temporal
{
    // Solves the equations $d\phi/dt = D(\nabla^2\phi - f(x, y))$
    template<typename T>
    struct Diffusion
    {
        double t = 0.0;
        const T D;

        Diffusion(T _D) :
            D(_D)
        {   }

        template<uint32_t N>
        void updateMesh(
            Mesh<T, N>& mesh,
            const std::function<T(const Eigen::Array<double, 1, 2>&, double)>& rhs,
            const double& dt)
        {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> func(mesh.space.rows(), mesh.space.cols());
            for (uint32_t y = 0; y < func.rows(); y++)
                for (uint32_t x = 0; x < func.cols(); x++)
                    func(y, x) = rhs(Eigen::Array<double, 1, 2>({ x, y }), t);

            decltype(func) diff(mesh.space.rows(), mesh.space.cols());
            diff = D * (laplacian(mesh) - func);

            for (uint32_t r = 0; r < diff.rows(); r++)
                for (uint32_t c = 0; c < diff.cols(); c++)
                    if (abs(diff(r, c)) <= 1e-7) diff(r, c) = 0;

            Mesh<T, N> new_mesh;
            new_mesh.space = mesh.space;
            new_mesh.space += diff * dt;

            decltype(func) diff2(mesh.space.rows(), mesh.space.cols());
            diff2 = D * (laplacian(new_mesh) - func);

            mesh.space += (diff + diff2) * dt / 2.0;
            t += dt;
        }
    };
}