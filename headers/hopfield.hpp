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
        void inicializarNeuronas(void (*f)(vector<Neurona>&)); //inicializa las neuronas segun la funcion f
        void cargarPatron(patron p); //carga un patron a la red
        void entrenar(); //entrenamiento de la red
        void calcularUmbrales(void (*f)(vector<float>&)); //calcula los umbrales segun la funcion f
        void calcularEnergia(); //calcula energia de la red
        float calcularEnergia(patron p); //calcula energia como si la red estuviera en un patron particular
        void evolucionar(); //evoluciona una neurona al azar
        float obtenerEnergia();
        vector<Neurona> obtenerNeuronas();
        vector<vector<float>> obtenerPesos();
        int cantidadNeuronas();
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

//FUNCIONES PARA TESTEAR Y ESTUDIAR EL FUNCIONAMIENTO DE LA RED

//funcion que evoluciona la red hasta que la energía cambie menos de 1%, devuelve la cantidad de epocas hasta este punto
int evolucionEnergia(Red& red, vector<float>& E);
//evoluciona la red n pasos y devuelve el error basado en patronAAlmacenar en el paso final y guarda el error en cada paso en errores
int evolucionErrores(Red&red, vector<int>& errores, int n, patron patronAAlmacenar);
//devuelve si tuvo exito a recuperar la memoria dados los patrones a almacenar
bool testearExito(Red& red, patron memoria, vector<patron> patronesAAlmacenar);