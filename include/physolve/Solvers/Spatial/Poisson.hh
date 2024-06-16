#pragma once

#include <optional>

#include "../../Mesh.hh"

namespace ps::Solvers::Spatial
{
    // Solver for the 2D Poisson equation
    template<typename T, uint32_t N>
    struct Poisson
    {
        template<uint32_t D, typename Type = T>
        using Vector = Eigen::Array<Type, 1, D>;

        using RHS_Function = std::function<T(const Vector<2>&)>;
        
        // Solver that takes an input mesh, boundary conditions and (optional) right hand side
        // that, when zero, reduces to Laplace's equation. The information is written out
        // to the mesh.
        void solve(
            Mesh<T, N>& mesh, 
            const RHS_Function& boundary,
            const std::optional<RHS_Function> func = std::nullopt)
        {
            Eigen::SparseMatrix<T> A(N*N, N*N);
            A.setZero();

            // Set up the equation Au = q
            // where u is the vector of solutions corresponding to the 
            // positions. That is, u[i] is the value at position v[i].
            Eigen::Array<Vector<2, double>, Eigen::Dynamic, 1> v(N*N);
            Eigen::Matrix<T, Eigen::Dynamic, 1> q(N*N);
            q.setZero(); v.setZero();

            for (uint32_t y = 0; y < N; y++)
                for (uint32_t x = 0; x < N; x++)
                    v(y * N + x) = { x, y };

            // This coupled with the -1.0 on the diagonal
            // makes this matrix represent the Poisson equation
            using pair = std::pair<Vector<2>, T>;
            const std::vector<pair> matching = {
                pair({  0, -1 }, 0.25),
                pair({  0,  1 }, 0.25),
                pair({  1,  0 }, 0.25),
                pair({ -1,  0 }, 0.25)
            };

            // Generate the matrix as well as fill out the q vector
            for (uint32_t n = 0; n < N*N; n++)
            {
                A.insert(n, v(n)(1) * N + v(n)(0)) = -1.0;

                for (const auto& p : matching)
                {
                    const auto point = v(n) + p.first;
                    if (point(0) > 0 && point(1) > 0 && point(0) < N - 1 && point(1) < N - 1)
                        A.insert(n, point(1) * N + point(0)) = p.second;
                    else
                        if (point(0) >= 0 && point(1) >= 0 && point(0) < N && point(1) < N)
                            q(n) += boundary(point) * p.second;
                }
            }
            A.makeCompressed();

            // If there's a rhs, take away from the q vector the values of the rhs
            // at each point
            if (func)
                for (uint32_t i = 0; i < N*N; i++)
                    q(i) -= func.value()(v(i));

            // Solve the system for the vector u and assign it to the mesh space
            Eigen::SimplicialLDLT<decltype(A)> solver(A);
            decltype(q) u = solver.solve(-1.0 * q);
            for (uint32_t y = 0; y < N; y++)
                for (uint32_t x = 0; x < N; x++)
                    mesh.space(y, x) = u(y * N + x);
        }
    };
}