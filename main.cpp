#include <iostream>

#include <physolve/physolve.hh>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace ps;
using namespace ps::Solvers::Spatial;
using namespace ps::Solvers::Temporal;

const auto SIZE = 150U;
const auto vel_mag = 100.f;

struct Particle
{
    Math::Vec2f position, velocity;
    double mass = 4.0;
};

struct Color
{
    uint8_t r, g, b;

    static friend Color operator-(Color a, Color b)
    {
        Color c;
        c.r = a.r - b.r;
        c.g = a.g - b.g;
        c.b = a.b - b.b;
        return c;
    }
    
    static friend Color operator+(Color a, Color b)
    {
        Color c;
        c.r = a.r + b.r;
        c.g = a.g + b.g;
        c.b = a.b + b.b;
        return c;
    }

    template<typename T>
    static friend Color operator*(Color a, T s)
    {
        Color c;
        c.r = a.r * s;
        c.g = a.g * s;
        c.b = a.b * s;
        return c;
    }
};

template<typename _Matrix>
void
writeMatrix(const std::string& filename, const _Matrix& matrix, std::optional<double> max = std::nullopt)
{
    //auto color = new Color[matrix.rows()][matrix.cols()];
    Color* color = new Color[matrix.rows() * matrix.cols()];
    
    const auto MAX_COEFF = matrix.maxCoeff();
    const auto MIN_COEFF = matrix.minCoeff();
    for (uint32_t y = 0; y < matrix.rows(); y++)
        for (uint32_t x = 0; x < matrix.cols(); x++)
        {
            const auto normalized = (matrix(y, x) - MIN_COEFF) / ((max.has_value()?max.value():MAX_COEFF) - MIN_COEFF);
            const auto colorb = Color{1, 0, 0};
            const auto colora = Color{0, 0, 1};
            const auto interp = (colorb - colora) * (float)normalized + colora;
            color[y * matrix.cols() + x] = interp;
        }

    stbi_write_bmp(filename.c_str(), matrix.cols(), matrix.rows(), 3, color);

    delete[] color;
}

int main()
{
    Mesh<double, SIZE> field;
    Poisson<double, SIZE> solver;
    Wave wave_solver(10.0, field);
    
    std::vector<Particle> particles;
    for (uint32_t i = 0; i < 50000; i++)
        particles.push_back({ 
            { rand() / (float)RAND_MAX * SIZE, rand() / (float)RAND_MAX * SIZE }, 
            { (rand() / (float)RAND_MAX - 0.5f) * vel_mag, (rand() / (float)RAND_MAX - 0.5f) * vel_mag}, 
            rand() / (float)RAND_MAX * 4.0 });

    const auto make_mass = [&particles](const auto& pos, double t)
    {
        auto cell = 0.0;
        for (const auto& part : particles)
            //if ((sf::Vector2i)part.position == sf::Vector2i(pos(0), pos(1)) && sf::FloatRect({ 0, 0 }, { SIZE - 1, SIZE - 1 }).contains(part.position))
            if ( (int)Math::x(part.position) == (int)pos(0) && (int)Math::y(part.position) == (int)pos(1) &&
                 Math::x(part.position) >= 0 && Math::x(part.position) <= SIZE - 1 &&
                 Math::y(part.position) >= 0 && Math::y(part.position) <= SIZE - 1 )
                    cell += part.mass;
        
        return cell;
    };

    const auto make_mass_0 = [&make_mass](const auto& pos) { return make_mass(pos, 0); };

    for (uint32_t it = 0; it < 100; it++)
    {
        std::cout << "it " << it << "\n";

        std::cout << "solving\n";
        solver.solve(field, [](const auto& pos) { return 0.0; }, make_mass_0);

        for (auto& p : particles)
        {
            if (!(Math::x(p.position) >= 0 && Math::x(p.position) <= SIZE - 1 &&
                  Math::y(p.position) >= 0 && Math::y(p.position) <= SIZE - 1 ))
            {
                p.position = { rand() / (float)RAND_MAX * SIZE, rand() / (float)RAND_MAX * SIZE };
                p.velocity = { (rand() / (float)RAND_MAX - 0.5f) * vel_mag, (rand() / (float)RAND_MAX - 0.5f) * vel_mag };
                p.mass = rand() / (double)RAND_MAX * 4.0;
            }

            const auto grad = gradient(field.space, Math::x(p.position), Math::y(p.position), SIZE);
            Math::x(p.velocity) += -4.f * p.mass * grad.x * 0.005;
            Math::y(p.velocity) += -4.f * p.mass * grad.y * 0.005;

            p.position += p.velocity * 0.005f;
        }

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mass(SIZE, SIZE);
        for (uint32_t r = 0; r < SIZE; r++)
            for (uint32_t c = 0; c < SIZE; c++)
                mass(r, c) = make_mass_0(Eigen::Array<double, 1, 2>({ c, r }));

        std::cout << "writing\n";
        writeMatrix((std::stringstream() << "frame-" << it << ".bmp").str(), mass);
    }

    return 0;
}