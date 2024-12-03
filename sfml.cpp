#include "headers/sfml.hpp"

void graficarPatron(RenderWindow& ventana, const std::vector<int>& patron, std::string guardarEn){
    ventana.clear(); // Limpia la ventana antes de dibujar

    int width = ventana.getSize().x / cellSize;
    int height = ventana.getSize().y / cellSize;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            
            RectangleShape cell(Vector2f(cellSize, cellSize));
            cell.setPosition(x * cellSize, y * cellSize);

            // Cambia el color según el valor de la neurona (-1 o 1)
            if(patron[index] == 1) {
                cell.setFillColor(Color::Green);
            }else {
                cell.setFillColor(Color::Red);
            }

            cell.setOutlineColor(Color::Black);
            cell.setOutlineThickness(2);

            ventana.draw(cell);
        }
    }

    if(guardarEn != ""){
        Texture texture;
        texture.create(ventana.getSize().x, ventana.getSize().y);
        texture.update(ventana);
        Image screenshot = texture.copyToImage();
        screenshot.saveToFile(guardarEn);
    }

    ventana.display(); // Muestra lo que se dibujo en la ventana
}

int cellSize2 = 5;
void graficarMatrizPesos(RenderWindow& ventana, const std::vector<std::vector<float>>& w){
    ventana.clear(); // Limpia la ventana antes de dibujar

    int width = ventana.getSize().x / cellSize2;
    int height = ventana.getSize().y / cellSize2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            RectangleShape cell(Vector2f(cellSize2, cellSize2));
            cell.setPosition(x * cellSize2, y * cellSize2);

            // Cambia el color segun la intensidad del peso
            int color = (w[y][x] + 1) * 127.5;
            cell.setFillColor(Color(color, color, color));

            cell.setOutlineColor(Color::Black);
            cell.setOutlineThickness(2);

            ventana.draw(cell);
        }
    }

    ventana.display(); // Muestra lo que se dibujo en la ventana

}