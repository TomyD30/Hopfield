#include "headers/sfml.hpp"

void displayNetwork(RenderWindow& window, const std::vector<int>& pattern) {
    window.clear(); // Limpia la ventana antes de dibujar

    int width = window.getSize().x / cellSize;
    int height = window.getSize().y / cellSize;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            
            RectangleShape cell(Vector2f(cellSize, cellSize));
            cell.setPosition(x * cellSize, y * cellSize);

            // Cambia el color según el valor de la neurona (-1 o 1)
            if (pattern[index] == 1) {
                cell.setFillColor(Color::Green);
            } else {
                cell.setFillColor(Color::Red);
            }

            cell.setOutlineColor(Color::Black);
            cell.setOutlineThickness(2);

            window.draw(cell);
        }
    }

    window.display(); // Muestra lo que se ha dibujado en la ventana
}
