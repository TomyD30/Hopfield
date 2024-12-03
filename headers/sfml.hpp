#include <SFML/Graphics.hpp>

using namespace sf;

const int cellSize = 25;
//grafica patron en la ventana y lo guarda en guardarEn si se quiere
void graficarPatron(RenderWindow& ventana, const std::vector<int>& patron, std::string guardarEn = "");
//grafica la matriz de pesos
void graficarMatrizPesos(RenderWindow& ventana, const std::vector<std::vector<float>>& w);