#include "hopfield.hpp"
#include "omp.h"

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

void Red::calcularUmbrales(void (*f)(vector<float>&th)){
    f(th);
}

void Red::calcularEnergia(){
    E = 0;
    #pragma omp parallel for reduction(+:E)
    for(int i = 0; i < N; i++){
        E += th[i]*neuronas[i];
        for(int j = 0; j < N; j++){
            E -= 0.5*w[i][j]*neuronas[i]*neuronas[j];
        }
    }
}

// void Red::evolucionar(){
//     do{
//         calcularEnergia();
//         int i = rand()%N;
//         float r = 0.0;
//         for(int j = 0; j < N; j++){
//             r+= w[i][j]*neuronas[j];
//         }
//         if(r >= th[i]) neuronas[i] = 1;
//         else neuronas[i] = -1;
//     }while(E < 1e-5); //esto no se si esta bien, que seria la condicion de estable? supuse E = 0
// }
void Red::evolucionar(){
    int i = rand()%N;
    float r = 0.0;
    for(int j = 0; j < N; j++){
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
}

float RedContinua::obtenerEnergia(){
    calcularEnergia();
    return E;
}

vector<NeuronaC> RedContinua::obtenerNeuronas(){
    return neuronas;
}