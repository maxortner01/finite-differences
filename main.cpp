#include <iostream>
#include <Eigen/Eigen>
#include <Eigen/Dense>

template<uint32_t N_>
struct Mesh
{
    using Real = double;
    constexpr static uint32_t N = N_;
    
    Eigen::Matrix<Real, N, N> space;

    Mesh()
    { space.setZero(); }
};

template<uint32_t N>
struct PoissonSolver
{
    using Real = typename Mesh<N>::Real;

    template<uint32_t D>
    using Vector = Eigen::Array<Real, 1, D>;

    Eigen::Matrix<Real, N*N, N*N> A;

    PoissonSolver()
    { A.setZero(); }

    void solve(
        Mesh<N>& mesh, 
        const std::function<Real(const Vector<2>&)>& boundary)
    {
        decltype(A) temp_A;
        temp_A.setZero();

        Eigen::Array<Vector<2>, N*N, 1> v;
        Eigen::Matrix<Real, N*N, 1> q;
        q.setZero();
        v.setZero();
        for (uint32_t y = 0; y < N; y++)
            for (uint32_t x = 0; x < N; x++)
                v(y * N + x) = { x, y };

        using pair = std::pair<Vector<2>, Real>;
        const std::vector<pair> matching = {
            pair({  0, -1 }, 0.25),
            pair({  0,  1 }, 0.25),
            pair({  1,  0 }, 0.25),
            pair({ -1,  0 }, 0.25)
        };

        for (uint32_t n = 0; n < N*N; n++)
        {
            temp_A(n, v(n)(1) * N + v(n)(0)) = -1.0;

            for (const auto& p : matching)
            {
                const auto point = v(n) + p.first;
                if (point(0) > 0 && point(1) > 0 && point(0) < N - 1 && point(1) < N - 1)
                    temp_A(n, point(1) * N + point(0)) = p.second;
                else
                    if (point(0) >= 0 && point(1) >= 0 && point(0) < N && point(1) < N)
                        q(n) += boundary(point) * p.second;
            }
        }

        A = temp_A.inverse();
        decltype(q) u = (A*(-1.0 * q));
        u.reshaped(N, N);
        std::cout << u << "\n";
    }
};

int main()
{
    Mesh<4> mesh;
    PoissonSolver<4> solver;
    solver.solve(mesh,
    [](const auto& pos) -> auto
    {
        if (pos(0) == 0) return 1.0;
        return 0.0;
    });
}