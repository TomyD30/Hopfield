#include "hopfield.hpp"
#include "mnist.hpp"
#include "sfml.hpp"
#include <fstream>
#include <SFML/Graphics.hpp>

//condiciones iniciales
void umbralesNulos(vector<float>& th);
void neuronasAlAzar(vector<Neurona>& neuronas);
void neuronasConError(vector<Neurona>& neuronas);
void umbralesPromedio(vector<float>& th);
void umbralesPromedioPorNeurona(vector<float>& th);
void umbralesPostEntreno(vector<float>& th);
void neuronasParciales(vector<Neurona>& neuronas);

//ejercicios
void ej1();
void ej2();
void ej3();
void ej4();

//extras
void probandoCosas();
void ej3_2();
void ej3_3();
void ej4_2();
void capacidadesGenerales();