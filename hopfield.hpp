#include <iostream>
#include <vector>

using namespace std;

using Neurona = int; // -1 o 1 (puedo crear una struct si es necesario)
using patron = vector<int>;

class Red{
    private:
        int N; //cantidad de neuronas
        vector<Neurona> neuronas;
        float E; //energia
        vector<vector<float>> w; //matriz de pesos
        vector<float> th; //vector de umbrales
        vector<patron> patrones; //patrones almacenados
        int M; //cantidad de patrones almacenados
    public:
        Red(int N);
        void inicializarNeuronas(void (*f)(vector<Neurona>&));
        void cargarPatron(patron p);
        void entrenar();
        void calcularUmbrales(void (*f)(vector<float>&));
        void calcularEnergia();
        void evolucionar();
        float obtenerEnergia();
        vector<Neurona> obtenerNeuronas();
};

//seguramente transforme este en una RedSimple y luego cree otra para la parte de meterle temperatura