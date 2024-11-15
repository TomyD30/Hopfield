#include <iostream>
#include <vector>
#include <cmath>

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
        vector<vector<float>> obtenerPesos();
};

//seguramente transforme este en una RedSimple y luego cree otra para la parte de meterle temperatura
//puedo usar templates para genelarizar la red para distintos tipos de neuronas

using NeuronaC = float;
using patronC = vector<float>;

class RedContinua{
    private:
        int N; //cantidad de neuronas
        vector<NeuronaC> neuronas;
        float E; //energia
        float T; //temperatura
        vector<vector<float>> w; //matriz de pesos
        vector<float> th; //vector de umbrales
        vector<patronC> patrones; //patrones almacenados
        int M; //cantidad de patrones almacenados
    public:
        RedContinua(int N, float T);
        void inicializarNeuronas(void (*f)(vector<NeuronaC>&));
        void cargarPatron(patronC p);
        void entrenar();
        void calcularUmbrales(void (*f)(vector<float>&));
        void calcularEnergia();
        void evolucionar();
        float obtenerEnergia();
        vector<NeuronaC> obtenerNeuronas();
};