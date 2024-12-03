#include "headers/hopfield.hpp"
#include <omp.h>

Red::Red(int N): N(N){
    neuronas = vector<Neurona>(N);
    w = vector<vector<float>>(N, vector<float>(N));
    th = vector<float>(N);
    M = 0;
    E = 0.0;
}

void Red::inicializarNeuronas(void (*f)(vector<Neurona>&)){
    f(neuronas);
}

void Red::cargarPatron(patron p){
    patrones.push_back(p);
    M++;
}

void Red::entrenar(){
    #pragma omp parallel for
    for(int i = 0; i < N; i++){ //entrena la red de forma paralela
        w[i][i] = 0;
        for(int j = i+1; j < N; j++){ //aprovecha la simetria de la matriz de pesos
            w[i][j] = 0;
            for(int m = 0; m < M; m++){
                w[i][j] += patrones[m][i]*patrones[m][j];
            }
            w[i][j] /= M;
            w[j][i] = w[i][j];
        }
    }
}

void Red::calcularUmbrales(void (*f)(vector<float>&th)){
    f(th);
}

void Red::calcularEnergia(){
    E = 0;
    #pragma omp parallel for reduction(+:E) //paralelizacion del calculo de la energia
    for(int i = 0; i < N; i++){
        E += th[i]*neuronas[i];
        for(int j = 0; j < N; j++){
            E += -0.5*w[i][j]*neuronas[i]*neuronas[j];
        }
    }
}
float Red::calcularEnergia(patron p){
    float E_ = 0;
    #pragma omp parallel for reduction(+:E_)
    for(int i = 0; i < N; i++){
        E_ += th[i]*p[i];
        for(int j = 0; j < N; j++){
            E_ += -0.5*w[i][j]*p[i]*p[j];
        }
    }
    return E_;
}

void Red::evolucionar(){
    int i = rand()%N; //eligo una neurona al azar
    float r = 0.0;
    for(int j = 0; j < N; j++){ //calculo del factor que determina la evolucion
        r += w[i][j]*neuronas[j];
    }
    if(r >= th[i]) neuronas[i] = 1;
    else neuronas[i] = -1;
}

float Red::obtenerEnergia(){
    calcularEnergia();
    return E;
}

vector<Neurona> Red::obtenerNeuronas(){
    return neuronas;
}

vector<vector<float>> Red::obtenerPesos(){
    return w;
}   

int Red::cantidadNeuronas(){
    return N;
}

// RED CONTINUA 

RedContinua::RedContinua(int N, float T): N(N), T(T){
    neuronas = vector<NeuronaC>(N);
    w = vector<vector<float>>(N, vector<float>(N));
    th = vector<float>(N);
    M = 0;
    E = 0.0;
}

void RedContinua::inicializarNeuronas(void (*f)(vector<NeuronaC>&)){
    f(neuronas);
}

void RedContinua::cargarPatron(patronC p){
    patrones.push_back(p);
    M++;
}

void RedContinua::entrenar(){
    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(i != j){
                for(int m = 0; m < M; m++){
                    #pragma omp atomic
                    w[i][j] += patrones[m][i]*patrones[m][j];
                }
                w[i][j] /= M;
            }
        }
    }
}

void RedContinua::calcularUmbrales(void (*f)(vector<float>&th)){
    f(th);
}

void RedContinua::calcularEnergia(){
    E = 0;
    #pragma omp parallel for reduction(+:E)
    for(int i = 0; i < N; i++){
        E += th[i]*neuronas[i];
        for(int j = 0; j < N; j++){
            E -= 0.5*w[i][j]*neuronas[i]*neuronas[j];
        }
    }
}

void RedContinua::evolucionar(){
    int i = rand()%N;
    float r = 0.0;
    for(int j = 0; j < N; j++){
        r += w[i][j]*neuronas[j];
    }
    neuronas[i] = tanh((r-th[i])/T);
    T *= 0.999;
}

float RedContinua::obtenerEnergia(){
    calcularEnergia();
    return E;
}

vector<NeuronaC> RedContinua::obtenerNeuronas(){
    return neuronas;
}

int evolucionEnergia(Red& red, vector<float>& E){
    int N = red.cantidadNeuronas();
    int n = N; //numero de evoluciones (al menos N)
    E.resize(n);
    for(int i = 0; i < n; i++){
        red.evolucionar();
        E.at(i) = red.obtenerEnergia();
        if((i+1)%N == 0){
            if(abs((E.at(i)-E.at(i+1-N))/E.at(i+1-N)) < 0.01) break; // si luego de N evoluciones, la energia cambia menos de 1% considero que es estable
            else n += N, E.resize(n); //si no, hago N evoluciones mas
        }
    }
    return n/N;
}

int evolucionErrores(Red&red, vector<int>& errores, int n, patron patronAAlmacenar){
    int N = red.cantidadNeuronas();
    int error = 0;
    for(int i = 0; i < n; i++){
        red.evolucionar();
        vector<Neurona> neuronas = red.obtenerNeuronas();
        int error_i = 0;
        for(int j = 0; j < N; j++){
            if(neuronas[j] != patronAAlmacenar[j]) error_i++;
        }
        if(i == n-1) error = error_i;
        errores[i] += error_i;
    }
    return error;
}

bool testearExito(Red& red, patron memoria, vector<patron> patronesAAlmacenar){
    int N = red.cantidadNeuronas();
    int n = N; //numero de evoluciones
    bool exito;
    int erroresEpoca = 0;
    vector<Neurona> neuronas = red.obtenerNeuronas();
    for(int j = 0; j < N; j++){ //calculo errores iniciales
        if(neuronas[j] != memoria[j]) erroresEpoca++;
    }
    for(int i = 0; i < n; i++){
        red.evolucionar();
        if(i == n-1){
            neuronas = red.obtenerNeuronas();
            int errores = 0;
            for(int j = 0; j < N; j++){
                if(neuronas[j] != memoria[j]) errores++;
            }
            if(errores < 0.05*N){ //si tiene menos de 5% de errores considero que tuvo exito
                exito = true;
                break;
            }
            else if(errores >= erroresEpoca){ // si la cantidad de errores se quedo igual o aumenta lo considero fracaso
                exito = false;
                break;
            }
            else erroresEpoca = errores, n += N; // si no pasa ninguna de estas dos cosas, aumento la cantidad de evoluciones y repito proceso
        }
    }
    return exito;
}
