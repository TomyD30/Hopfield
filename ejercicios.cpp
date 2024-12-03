#include "headers/ejercicios.hpp"

//1. Comenzando de condiciones iniciales al azar para las neuronas analice la evolución de la energía en función de los pasos de evolución de la red.
void umbralesNulos(vector<float>& th){
    for(int i = 0; i < th.size(); i++){
        th[i] = 0;
    }
}
void neuronasAlAzar(vector<Neurona>& neuronas){
    for(int i = 0; i < neuronas.size(); i++){
        neuronas[i] = 2*(rand()%2)-1;
    }
}
void ej1(){
    srand(time(NULL));

    int N = 256;
    patron p; //entreno la red con un patron aleatorio
    for(int i = 0; i < N; i++){
        p.push_back(2*(rand()%2)-1);
    }
    for(int k = 0; k < 4; k++){ //realizo 4 iteraciones con condiciones iniciales distintas
        Red red(N);
        red.inicializarNeuronas(neuronasAlAzar);
        red.cargarPatron(p);
        red.entrenar();
        red.calcularUmbrales(umbralesNulos);

        vector<float> E;
        int e = evolucionEnergia(red,E);
        cout << "Iteracion " << k << " convergio en " << e << " epocas" << endl;
        ofstream archivo;
        archivo.open("Resultados/E_N" + to_string(N) + "(" + to_string(k+1) +  ").txt");
        for(int i = 0; i < E.size(); i++){
            archivo << E.at(i) << endl;
        }
        archivo.close();
    }
}

//2. Partiendo de uno de los patrones a almacenar con algún número de modificaciones aleatorias, analice la convergencia de la red al patrón original, calculando el número de neuronas que difieren del patron original.
patron patronAAlmacenar;
float ruido; // [0,1]
void neuronasConError(vector<Neurona>& neuronas){ //pone todas las neuronas = al patron a almacenar y un porcentaje ruido% se invierte
    for(int i = 0; i < neuronas.size(); i++){
        neuronas[i] = patronAAlmacenar[i];
        if(1.0*rand()/RAND_MAX < ruido) neuronas[i] *= -1;
    }
}
const string imagenesMNIST = "train-images.idx3-ubyte";
int numImagenes, numFilas, numCols;
void ej2(){
    srand(time(NULL));

    auto imagenes = cargarImagenesMNIST(imagenesMNIST, numImagenes, numFilas, numCols);
    patronAAlmacenar = binarizarImagen(imagenes[0]);

    int N = numFilas*numCols;
    int n = 5*N; //numero de evoluciones por iteracion
    int it = 100; //numero de iteraciones
    vector<int> erroresPorRuido;
    for(float r = 0.1; r < 1.0; r += 0.05){ //pruebo errores desde 10% a 95% con pasos de 5%
        int error = 0;
        vector<int> errores;
        errores.resize(n);
        for(int k = 0; k < it; k++){ // itero it veces sobre el mismo ruido
            ruido = r;
            Red red(N);
            red.inicializarNeuronas(neuronasConError);
            red.cargarPatron(patronAAlmacenar);
            red.entrenar();
            red.calcularUmbrales(umbralesNulos);

            error += evolucionErrores(red,errores,n,patronAAlmacenar);
        }
        erroresPorRuido.push_back(error/it);
        ofstream archivo;
        archivo.open("Resultados/err_r" + to_string(r) + ".txt");
        for(int i = 0; i < errores.size(); i++){
            archivo << errores[i]/it << endl;
        }
        archivo.close();
        cout << "r: " << r << " | err: " << error/it << endl;
    }
    ofstream archivo;
    archivo.open("Resultados/errores.txt");
    for(int i = 0; i < erroresPorRuido.size(); i++){
        archivo << erroresPorRuido[i] << endl;
    }
    archivo.close();
}

//3.Analice la capacidad de almacenamiento de la red. Para ello estudie el éxito al recuperar una memoria en función del número M de patrones almacenados. 
//Tenga en cuenta que el estado al que converge la red puede no diferir de los patrones de entrenamiento.
vector<patron> patronesAAlmacenar;
void ej3(){
    srand(time(NULL));
    
    auto imagenes = cargarImagenesMNIST(imagenesMNIST, numImagenes, numFilas, numCols);

    int L = 14;
    int N = L*L;
    ruido = 0.1;
    vector<int> capacidades;
    int iteraciones = 1000;

    // sf::RenderWindow window(sf::VideoMode(L * cellSize, L * cellSize), "Red de Hopfield");

    for(int k = 0; k < iteraciones; k++){
        if(k%100 == 0) cout << k << endl;
        patronesAAlmacenar.clear();
        int Mmax = 2;
        Red red(N);
        for(int M = 1; M < Mmax; M++){
            for(int m = patronesAAlmacenar.size(); m < M; m++){
                patron imagenBinaria = binarizarImagen(imagenes[rand()%numImagenes]);
                // patron imagenBinaria = binarizarImagen(imagenes[m]);
                // editarImagen2(imagenBinaria,numFilas,numCols);
                // patronesAAlmacenar.push_back(imagenBinaria);
                // displayNetwork(window,patronesAAlmacenar[m]);
                // sf::sleep(sf::seconds(2));
                patronesAAlmacenar.push_back(cortarImagen(imagenBinaria,L,numFilas,numCols));
                red.cargarPatron(patronesAAlmacenar[m]);
            }
            patronAAlmacenar = patronesAAlmacenar[0];
            red.inicializarNeuronas(neuronasConError);
            red.calcularUmbrales(umbralesNulos);
            red.entrenar();
            // sf::RenderWindow window2(sf::VideoMode(N * 5, N * 5), "Matriz de pesos");
            // graficarMatrizPesos(window2,red.obtenerPesos());
            // sf::sleep(sf::seconds(1000));
            // window2.close();

            bool exito = testearExito(red,patronAAlmacenar,patronesAAlmacenar);
            if(exito) Mmax++; //si tuvo exito intento almacenar un patron mas
            else capacidades.push_back(M-1); //si no tuvo exito guardo la capacidad maxima de la red
        }
    }
    float promedio = 0;
    for(int i = 0; i < capacidades.size(); i++){
        promedio += capacidades[i];
    }
    promedio /= capacidades.size();
    cout << "Capacidad de almacenamiento promedio: " << promedio << endl;
    ofstream archivo;
    archivo.open("Resultados/cap_N" + to_string(N) + ".txt");
    for(int i = 0; i < capacidades.size(); i++){
        archivo << capacidades[i] << endl;
    }
}

//4. Considere usar θi=θ¯ para los umbrales.
void umbralesPromedio(vector<float>& th){
    int N = th.size();
    int M = patronesAAlmacenar.size();
    float promedio = 0;
    for(int i = 0; i < N; i++){
        for(int k = 0; k < M; k++){
            promedio += patronesAAlmacenar[k][i];
        }
    }
    promedio /= M*N;
    for(int i = 0; i < N; i++){
        th[i] = -promedio;
    }
}
void umbralesPromedioPorNeurona(vector<float>& th){
    int N = th.size();
    int M = patronesAAlmacenar.size();
    for(int i = 0; i < N; i++){
        float promedio = 0;
        for(int k = 0; k < M; k++){
            promedio += patronesAAlmacenar[k][i];
        }
        promedio /= M;
        th[i] = promedio;
    }
}
void ej4(){
    srand(time(NULL));

    auto imagenes = cargarImagenesMNIST(imagenesMNIST, numImagenes, numFilas, numCols);
    patronAAlmacenar = binarizarImagen(imagenes[0]);
    patronesAAlmacenar.push_back(patronAAlmacenar);

    string carpeta = "Resultados/umbrales_promedio/";

    int L = 28;
    int N = L*L;

    // //repito ej1
    // for(int k = 0; k < 4; k++){
    //     Red red(N);
    //     red.inicializarNeuronas(neuronasAlAzar);
    //     red.cargarPatron(patronAAlmacenar);
    //     red.entrenar();
    //     red.calcularUmbrales(umbralesPromedio);

    //     vector<float> E;
    //     int e = evolucionEnergia(red,E);
    //     cout << "Iteracion " << k << " convergio en " << e << " epocas" << endl;
    //     ofstream archivo;
    //     archivo.open("Resultados/umbrales_promedio/E_N" + to_string(N) + "(" + to_string(k+1) +  ").txt");
    //     for(int i = 0; i < E.size(); i++){
    //         archivo << E.at(i) << endl;
    //     }
    //     archivo.close();
    // }

    // //repito ej2
    // int n = 2*N; //numero de evoluciones por iteracion
    // int it = 100; //numero de iteraciones
    // vector<int> erroresPorRuido;
    // for(float r = 0.1; r < 1.0; r += 0.05){
    //     int error = 0;
    //     vector<int> errores;
    //     errores.resize(n);
    //     for(int k = 0; k < it; k++){
    //         ruido = r;
    //         Red red(N);
    //         red.inicializarNeuronas(neuronasConError);
    //         red.cargarPatron(patronAAlmacenar);
    //         red.entrenar();
    //         red.calcularUmbrales(umbralesPromedio);

    //         error += evolucionErrores(red,errores,n,patronAAlmacenar);
    //     }
    //     erroresPorRuido.push_back(error/10);
    //     ofstream archivo;
    //     archivo.open("Resultados/umbrales_promedio/err_r" + to_string(r) + ".txt");
    //     for(int i = 0; i < errores.size(); i++){
    //         archivo << errores[i]/10 << endl;
    //     }
    //     archivo.close();
    // }
    // ofstream archivo;
    // archivo.open("Resultados/umbrales_promedio/errores.txt");
    // for(int i = 0; i < erroresPorRuido.size(); i++){
    //     archivo << erroresPorRuido[i] << endl;
    // }
    // archivo.close();

    //repito ej3
    ruido = 0.1;
    vector<int> capacidades;
    int iteraciones = 1000;
    for(int k = 0; k < iteraciones; k++){
        patronesAAlmacenar.clear();
        int Mmax = 2;
        Red red(N);
        for(int M = 1; M < Mmax; M++){
            for(int m = patronesAAlmacenar.size(); m < M; m++){
                patron imagenBinaria = binarizarImagen(imagenes[rand()%numImagenes]);
                patronesAAlmacenar.push_back(cortarImagen(imagenBinaria,L,numFilas,numCols));
                red.cargarPatron(patronesAAlmacenar[m]);
            }
            patronAAlmacenar = patronesAAlmacenar[0];
            red.inicializarNeuronas(neuronasConError);
            red.calcularUmbrales(umbralesPromedio);
            red.entrenar();
            // sf::RenderWindow window2(sf::VideoMode(N * 5, N * 5), "Matriz de pesos");
            // graficarMatrizPesos(window2,red.obtenerPesos());
            // sf::sleep(sf::seconds(1000));
            // window2.close();

            bool exito = testearExito(red,patronAAlmacenar,patronesAAlmacenar);
            if(exito) Mmax++;
            else capacidades.push_back(M-1);
        }
    }
    float promedio = 0;
    for(int i = 0; i < capacidades.size(); i++){
        promedio += capacidades[i];
    }
    promedio /= capacidades.size();
    cout << "Capacidad de almacenamiento promedio: " << promedio << endl;
    ofstream archivo;
    archivo.open(carpeta + "cap_N" + to_string(N) + ".txt");
    for(int i = 0; i < capacidades.size(); i++){
        archivo << capacidades[i] << endl;
    }
    archivo.close();
}

//5. Analice la posibilidad de modificar la regla de activación a: σi=tanh(∑jwijσj−θiT) , donde  T  es una temperatura efectiva y los  σi  son ahora continuos y toman valores en el intervalo  [−1,1] .
void displayNetwork(sf::RenderWindow& window, const vector<float>& pattern) {
    window.clear(); // Limpia la ventana antes de dibujar

    int width = window.getSize().x / cellSize;
    int height = window.getSize().y / cellSize;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            
            sf::RectangleShape cell(sf::Vector2f(cellSize, cellSize));
            cell.setPosition(x * cellSize, y * cellSize);

            //Cambia el color de la neurona en escala de grises
            int color = (pattern[index] + 1) * 127.5;
            cell.setFillColor(sf::Color(color, color, color));

            cell.setOutlineColor(sf::Color::Black);
            cell.setOutlineThickness(2);

            window.draw(cell);
        }
    }

    window.display(); // Muestra lo que se ha dibujado en la ventana
}
std::vector<float> imagenContinua(const std::vector<uint8_t>& image) {
    std::vector<float> imagenContinua(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        imagenContinua[i] = (image.at(i) - 127.5) / 127.5;
    }
    return imagenContinua;
}

patronC patronAAlmacenarC;
void neuronasConErrorC(vector<NeuronaC>& neuronas){
    for(int i = 0; i < neuronas.size(); i++){
        neuronas[i] = patronAAlmacenarC[i];
        if(rand()%100 < 10) neuronas[i] += 0.1*(2*(rand()%2)-1);
    }
}
vector<patronC> patronesAAlmacenarC;
void ej5(){ //no consegui nada interesante de esto, me gustaria probar mas
    srand(time(NULL));

    auto imagenes = cargarImagenesMNIST(imagenesMNIST, numImagenes, numFilas, numCols);

    sf::RenderWindow window(sf::VideoMode(numFilas * cellSize, numCols * cellSize), "Red Continua");
    for(int i = 0; i < 2; i++){
        patronAAlmacenarC = imagenContinua(imagenes[i]);
        patronesAAlmacenarC.push_back(patronAAlmacenarC);
        displayNetwork(window,patronAAlmacenarC);
    }
    patronAAlmacenarC = patronesAAlmacenarC[0];

    int N = numFilas*numCols;
    RedContinua red(N, 1);
    red.inicializarNeuronas(neuronasConErrorC);
    int M = patronesAAlmacenarC.size();
    for(int i = 0; i < M; i++){
        red.cargarPatron(patronesAAlmacenarC[i]);
    }
    red.entrenar();
    red.calcularUmbrales(umbralesNulos);

    displayNetwork(window,patronAAlmacenarC);
    sf::sleep(sf::seconds(1));
    displayNetwork(window,red.obtenerNeuronas());
    sf::sleep(sf::seconds(1));

    for(int i = 0; i < 10*N; i++){
        // sf::Event event;
        // while (window.pollEvent(event)) {
        //     if (event.type == sf::Event::Closed)
        //         window.close();
        // }
        red.evolucionar();
        // vector<NeuronaC> neuronas = red.obtenerNeuronas();
        // displayNetwork(window,neuronas);
        // sf::sleep(sf::milliseconds(10));
    }

    displayNetwork(window,red.obtenerNeuronas());
    sf::sleep(sf::seconds(1));
    
}

//aca estoy probando que pasa si entreno la red con mezcla de patrones de MNIST y patrones aleatorios
void probandoCosas(){
    srand(time(NULL));
    
    auto imagenes = cargarImagenesMNIST(imagenesMNIST, numImagenes, numFilas, numCols);

    int L = 28;
    int N = L*L;
    ruido = 0.1;
    vector<int> capacidades;
    int iteraciones = 1000;

    sf::RenderWindow window(sf::VideoMode(L * cellSize, L * cellSize), "Red de Hopfield");

    patron aux = binarizarImagen(imagenes[0]);
    patronAAlmacenar = cortarImagen(aux,L,numFilas,numCols);
    patronesAAlmacenar.push_back(patronAAlmacenar);
    for(int m = 0; m < 2; m++){
        patron imagenBinaria = binarizarImagen(imagenes[rand()%numImagenes]);
        patronesAAlmacenar.push_back(cortarImagen(imagenBinaria,L,numFilas,numCols));  
        // graficarPatron(window,patronesAAlmacenar[m+1]);
        // sf::sleep(sf::seconds(1));
    } 

    Red red(N);
    for(int m = 0; m < 10; m++){
        patron p(N);
        for(int i = 0; i < N; i++) p[i] = 2*(rand()%2)-1;
        patronesAAlmacenar.push_back(p);
    }
    for(int m = 0; m < patronesAAlmacenar.size(); m++) red.cargarPatron(patronesAAlmacenar[m]);
    red.inicializarNeuronas(neuronasConError);
    red.entrenar();
    red.calcularUmbrales(umbralesNulos);

    graficarPatron(window,patronAAlmacenar);
    sf::sleep(sf::seconds(1));
    graficarPatron(window,red.obtenerNeuronas());
    sf::sleep(sf::seconds(1));

    for(int k = 0; k < 10000; k++){
        red.evolucionar();
    }
    graficarPatron(window,red.obtenerNeuronas());
    sf::sleep(sf::seconds(1));
}

//aca estoy probando que pasa si entreno la red con mezcla de patrones de MNIST y cuando no puede almacenar mas le agrego patrones aleatorios
void ej3_2(){
    srand(time(NULL));
    
    auto imagenes = cargarImagenesMNIST(imagenesMNIST, numImagenes, numFilas, numCols);

    int L = 12;
    int N = L*L;
    ruido = 0.1;
    vector<int> capacidades;
    int iteraciones = 100;

    for(int k = 0; k < iteraciones; k++){
        if(k%10 == 0) cout << k << endl;
        bool exitoT = true; //exito al almacenar los patrones solo de MNIST
        patronesAAlmacenar.clear();
        int Mmax = 2;
        Red red(N);
        for(int M = 1; M < Mmax; M++){
            for(int m = patronesAAlmacenar.size(); m < M; m++){
                if(exitoT){
                    patron imagenBinaria = binarizarImagen(imagenes[rand()%numImagenes]);
                    patron p = cortarImagen(imagenBinaria,L,numFilas,numCols);
                    patronesAAlmacenar.push_back(p);
                }
                else{
                    patron p(N);
                    for(int i = 0; i < N; i++) p[i] = 2*(rand()%2)-1;
                    patronesAAlmacenar.push_back(p);
                }
                red.cargarPatron(patronesAAlmacenar[m]);
            }
            patronAAlmacenar = patronesAAlmacenar[0];
            red.inicializarNeuronas(neuronasConError);
            red.calcularUmbrales(umbralesNulos);
            red.entrenar();

            bool exito = testearExito(red,patronAAlmacenar,patronesAAlmacenar);
            if(exito) Mmax++;
            else if(exitoT) exitoT = false, Mmax++;
            else capacidades.push_back(M-1);
        }
    }
    float promedio = 0;
    for(int i = 0; i < capacidades.size(); i++){
        promedio += capacidades[i];
    }
    promedio /= capacidades.size();
    cout << "Capacidad de almacenamiento promedio: " << promedio << endl;
    ofstream archivo;
    archivo.open("Resultados/cap_N" + to_string(N) + "_ruidoExterno.txt");
    for(int i = 0; i < capacidades.size(); i++){
        archivo << capacidades[i] << endl;
    }
}

//aca estoy probando un solo patron de MNIST y todo el resto aleatorios
void ej3_3(){
    srand(time(NULL));
    
    auto imagenes = cargarImagenesMNIST(imagenesMNIST, numImagenes, numFilas, numCols);

    int L = 28;
    int N = L*L;
    ruido = 0.1;
    vector<int> capacidades;
    int iteraciones = 1000;

    for(int k = 0; k < iteraciones; k++){
        if(k%100 == 0) cout << k << endl;
        patronesAAlmacenar.clear();
        int Mmax = 60;
        Red red(N);
        patron imagenBinaria = binarizarImagen(imagenes[rand()%numImagenes]);
        patron p = cortarImagen(imagenBinaria,L,numFilas,numCols);
        patronesAAlmacenar.push_back(p);
        patronAAlmacenar = patronesAAlmacenar[0];
        red.cargarPatron(patronAAlmacenar);
        for(int M = Mmax-1; M < Mmax; M++){
            for(int m = patronesAAlmacenar.size(); m < M; m++){
                patron p(N);
                for(int i = 0; i < N; i++) p[i] = 2*(rand()%2)-1;
                patronesAAlmacenar.push_back(p);
                red.cargarPatron(patronesAAlmacenar[m]);
            }
            red.inicializarNeuronas(neuronasConError);
            red.calcularUmbrales(umbralesNulos);
            red.entrenar();

            bool exito = testearExito(red,patronAAlmacenar,patronesAAlmacenar);
            if(exito) Mmax++;
            else capacidades.push_back(M-1);
        }
    }
    float promedio = 0;
    for(int i = 0; i < capacidades.size(); i++){
        promedio += capacidades[i];
    }
    promedio /= capacidades.size();
    cout << "Capacidad de almacenamiento promedio: " << promedio << endl;
    ofstream archivo;
    archivo.open("Resultados/cap_N" + to_string(N) + "_ruidoExterno2.txt");
    for(int i = 0; i < capacidades.size(); i++){
        archivo << capacidades[i] << endl;
    }
}

//tercera tecnica de umbral: subirle la energia a los patrones espurios y disminuir la de los que me interesan
vector<patron> patronesEspurios;
void umbralesPostEntreno(vector<float>& th){
    int N = th.size();
    int M_ = patronesEspurios.size();
    int M = patronesAAlmacenar.size();
    for(int i = 0; i < N; i++){
        if(M_ == 0) th[i] = 0;
        else{
            for(int j = 0 ; j < M_; j++){// le subo la energia a los patrones espurios
                th[i] += patronesEspurios[j][i];
            }
            th[i] /= M_;
            for(int j = 0 ; j < M; j++){// le bajo la energia a los patrones que me interesan
                th[i] -= patronesAAlmacenar[j][i];
            }
            th[i] /= M;
        }
    }
}
//calcular los umbrales post entreno
void ej4_2(){
    srand(time(NULL));

    // sf::RenderWindow window(sf::VideoMode(10 * cellSize, 10 * cellSize), "Red de Hopfield");
    
    auto imagenes = cargarImagenesMNIST(imagenesMNIST, numImagenes, numFilas, numCols);

    int L = 28; //L TIENE QUE SER PAR!
    int N = L*L;
    ruido = 0.1;
    vector<int> capacidades;
    int iteraciones = 500;

    for(int k = 0; k < iteraciones; k++){
        if(k%100 == 0) cout << k << endl;
        patronesAAlmacenar.clear();
        patronesEspurios.clear();
        int Mmax = 2;
        Red red(N);

        patron imagenBinaria = binarizarImagen(imagenes[rand()%numImagenes]);
        patron p = cortarImagen(imagenBinaria,L,numFilas,numCols);
        patronesAAlmacenar.push_back(p);
        red.cargarPatron(patronesAAlmacenar[0]);
        patronAAlmacenar = patronesAAlmacenar[0];

        int intentos = 0; //intentos de subirle la energia a los espurios

        for(int M = 1; M < Mmax; M++){
            for(int m = patronesAAlmacenar.size(); m < M; m++){
                patron imagenBinaria = binarizarImagen(imagenes[rand()%numImagenes]);
                patron p = cortarImagen(imagenBinaria,L,numFilas,numCols);
                patronesAAlmacenar.push_back(p);
                red.cargarPatron(patronesAAlmacenar[m]);
            }
            red.inicializarNeuronas(neuronasConError);
            red.calcularUmbrales(umbralesPostEntreno);
            red.entrenar();

            bool exito = testearExito(red,patronAAlmacenar,patronesAAlmacenar);
            if(exito) Mmax++;
            else if(!exito and intentos < 10){
                patronesEspurios.push_back(red.obtenerNeuronas());
                M--;
                intentos++;
            }
            else capacidades.push_back(M-1);
        }
    }
    float promedio = 0;
    for(int i = 0; i < capacidades.size(); i++){
        promedio += capacidades[i];
    }
    promedio /= capacidades.size();
    cout << "Capacidad de almacenamiento promedio: " << promedio << endl;
    ofstream archivo;
    archivo.open("Resultados/umbralesPostEntreno/cap_N" + to_string(N) + ".txt");
    for(int i = 0; i < capacidades.size(); i++){
        archivo << capacidades[i] << endl;
    }
}

//aca estoy analizando capacidades generales de la red, como almacenar patrones a partir de una version parcial del mismo
void neuronasParciales(vector<Neurona>& neuronas){
    //solo mantiene la mitad de las neuronas
    for(int i = 0; i < neuronas.size(); i++){
        if(i < neuronas.size()/2) neuronas[i] = patronAAlmacenar[i];
        else neuronas[i] = -1;
    }
}
void capacidadesGenerales(){
    srand(time(NULL));

    auto imagenes = cargarImagenesMNIST(imagenesMNIST, numImagenes, numFilas, numCols);

    sf::RenderWindow window(sf::VideoMode(numFilas*cellSize, numCols*cellSize), "Red de Hopfield");

    int N = numFilas*numCols;
    ruido = 0.2;

    Red red(N);
    patronAAlmacenar = binarizarImagen(imagenes[7]);
    red.cargarPatron(patronAAlmacenar);
    red.cargarPatron(binarizarImagen(imagenes[0]));

    red.inicializarNeuronas(neuronasParciales);
    red.entrenar();
    red.calcularUmbrales(umbralesNulos);

    // graficarPatron(window,patronAAlmacenar,"Resultados/capacidadGeneral/patronOriginal.png");
    // sf::sleep(sf::seconds(1));
    // graficarPatron(window,binarizarImagen(imagenes[0]),"Resultados/capacidadGeneral/patron2.png");
    // sf::sleep(sf::seconds(1));

    graficarPatron(window,red.obtenerNeuronas(),"Resultados/capacidadGeneral/patronInicial_4.png");
    sf::sleep(sf::seconds(1));

    for(int i = 0; i < 5*N; i++){
        red.evolucionar();
    }
    graficarPatron(window,red.obtenerNeuronas(),"Resultados/capacidadGeneral/patronFinal_4.png");
    sf::sleep(sf::seconds(1));
}
