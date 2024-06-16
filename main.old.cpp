#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "./include/physolve/physolve.hh"

using namespace ps;
using namespace ps::Solvers::Spatial;
using namespace ps::Solvers::Temporal;

// Simple utility that conversts a matrix to an SFML bitmap
template<typename _Matrix>
void
matrixToImage(sf::Image& image, const _Matrix& matrix, bool create = true, std::optional<double> max = std::nullopt)
{
    if (create)
        image.create({ static_cast<uint32_t>(matrix.cols()), static_cast<uint32_t>(matrix.rows()) });
    else
        assert(sf::Vector2u(matrix.cols(), matrix.rows()) == image.getSize());
    
    const auto MAX_COEFF = matrix.maxCoeff();
    const auto MIN_COEFF = matrix.minCoeff();
    for (uint32_t y = 0; y < matrix.rows(); y++)
        for (uint32_t x = 0; x < matrix.cols(); x++)
        {
            const auto normalized = (matrix(y, x) - MIN_COEFF) / ((max.has_value()?max.value():MAX_COEFF) - MIN_COEFF);
            const auto colorb = sf::Vector3f(1, 0, 0);
            const auto colora = sf::Vector3f(0, 0, 1);
            const auto interp = (colorb - colora) * (float)normalized + colora;
            const auto color = sf::Color(
                (uint8_t)(interp.x * 255.f),
                (uint8_t)(interp.y * 255.f),
                (uint8_t)(interp.z * 255.f),
                255
            );
            image.setPixel({ x, y }, color);
        }
}

const auto SIZE = 250U;

struct Particle
{
    sf::Vector2f position, velocity;
    double mass = 4.0;
};

const auto vel_mag = 100.f;

int main()
{
    sf::RenderWindow window(sf::VideoMode({ 1280, 720 }), "Hello");

    sf::Image image;
    image.create({ SIZE, SIZE });

    Mesh<double, SIZE> field;
    Poisson<double, SIZE> solver;
    Wave wave_solver(10.0, field);

    std::vector<Particle> particles;
    for (uint32_t i = 0; i < 50000; i++)
        particles.push_back({ 
            sf::Vector2f(rand() / (float)RAND_MAX * SIZE, rand() / (float)RAND_MAX * SIZE), 
            sf::Vector2f((rand() / (float)RAND_MAX - 0.5f) * vel_mag, (rand() / (float)RAND_MAX - 0.5f) * vel_mag), 
            rand() / (float)RAND_MAX * 4.0 });

    for (const auto& p : particles) std::cout << p.position.x << ", " << p.position.y << "\n";

    const auto make_mass = [&particles](const auto& pos, double t)
    {
        auto cell = 0.0;
        for (const auto& part : particles)
            if ((sf::Vector2i)part.position == sf::Vector2i(pos(0), pos(1)) && sf::FloatRect({ 0, 0 }, { SIZE - 1, SIZE - 1 }).contains(part.position))
                cell += part.mass;
        
        return cell;
    };

    const auto make_mass_0 = [&make_mass](const auto& pos) { return make_mass(pos, 0); };

    uint32_t frame = 0;
    sf::Clock fps;
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) window.close();
        }

        window.clear();
        
        solver.solve(field, [](const auto& pos) { return 0.0; }, make_mass_0);

        //wave_solver.updateMesh(field, make_mass, 0.01);
        
        for (auto& p : particles)
        {
            if (!sf::FloatRect({ 0, 0 }, { SIZE - 1, SIZE - 1 }).contains(p.position))
            {
                p.position = sf::Vector2f(rand() / (float)RAND_MAX * SIZE, rand() / (float)RAND_MAX * SIZE);
                p.velocity = sf::Vector2f((rand() / (float)RAND_MAX - 0.5f) * vel_mag, (rand() / (float)RAND_MAX - 0.5f) * vel_mag);
                p.mass = rand() / (double)RAND_MAX * 4.0;
            }

            const auto grad = gradient(field.space, p.position.x, p.position.y, SIZE);
            p.velocity.x += -4.f * p.mass * grad.x * 0.005;
            p.velocity.y += -4.f * p.mass * grad.y * 0.005;

            p.position += p.velocity * 0.005f;
        }

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mass(SIZE, SIZE);
        for (uint32_t r = 0; r < SIZE; r++)
            for (uint32_t c = 0; c < SIZE; c++)
                mass(r, c) = make_mass_0(Eigen::Array<double, 1, 2>({ c, r }));

        matrixToImage(image, mass, false);
        std::cout << frame << "\n";
        image.saveToFile((std::stringstream() << "frame-" << frame++ << ".png").str());

        sf::Texture texture;
        assert(texture.loadFromImage(image));

        const auto scale = (float)window.getSize().y / (float)texture.getSize().y;

        sf::Sprite sprite(texture);
        sprite.setScale({ scale, scale });
        sprite.setPosition(sf::Vector2f(window.getSize().x / 2.f - sprite.getGlobalBounds().width / 2.f, 0));
        window.draw(sprite);

        window.display();

        window.setTitle((std::stringstream() << std::fixed << std::setprecision(2) << 1.f / fps.restart().asSeconds()).str());
    }
}