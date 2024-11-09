#include "hopfield.hpp"
#include <fstream>
#include <SFML/Graphics.hpp>
// #include <opencv2/opencv.hpp> //CREO QUE ACA HAY ERROR YA QUE DESCARGE VC16 QUE ES LA VERSION DE VISUAL, NECESITO LA DE MINGW

// std::vector<int> processImage(const std::string& filePath, int& width, int& height) {
//     // Cargar la imagen en escala de grises
//     cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
//     if (image.empty()) {
//         throw std::runtime_error("No se pudo abrir la imagen.");
//     }
//     // Cambiar el tamaño a 28x28 si es necesario
//     cv::resize(image, image, cv::Size(28, 28));
//     width = image.cols;
//     height = image.rows;

//     // Convertir a valores binarios
//     std::vector<int> binaryImage(width * height);
//     for (int y = 0; y < height; ++y) {
//         for (int x = 0; x < width; ++x) {
//             int pixel = image.at<uint8_t>(y, x); // Valor del píxel (0-255)
//             binaryImage[y * width + x] = (pixel > 128) ? 1 : -1; // Umbral en 128
//         }
//     }

//     return binaryImage;
// }

// Lee el encabezado de las imágenes MNIST y devuelve un vector de imágenes
std::vector<std::vector<uint8_t>> readMNISTImages(const std::string& filePath, int& numImages, int& numRows, int& numCols) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo de imágenes MNIST.");
    }

    int magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    magicNumber = __builtin_bswap32(magicNumber);
    if (magicNumber != 2051) {
        throw std::runtime_error("Número mágico inválido en el archivo de imágenes MNIST.");
    }

    file.read(reinterpret_cast<char*>(&numImages), 4);
    numImages = __builtin_bswap32(numImages);

    file.read(reinterpret_cast<char*>(&numRows), 4);
    numRows = __builtin_bswap32(numRows);

    file.read(reinterpret_cast<char*>(&numCols), 4);
    numCols = __builtin_bswap32(numCols);

    std::vector<std::vector<uint8_t>> images(numImages, std::vector<uint8_t>(numRows * numCols));
    for (int i = 0; i < numImages; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), numRows * numCols);
    }

    file.close();
    return images;
}

std::vector<int> binarizeImage(const std::vector<uint8_t>& image) {
    std::vector<int> binaryImage(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        binaryImage[i] = (image[i] > 128) ? 1 : -1;
    }
    return binaryImage;
}

const int cellSize = 25;
void displayNetwork(sf::RenderWindow& window, const std::vector<int>& pattern) {
    window.clear(); // Limpia la ventana antes de dibujar

    int width = window.getSize().x / cellSize;
    int height = window.getSize().y / cellSize;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            
            sf::RectangleShape cell(sf::Vector2f(cellSize, cellSize));
            cell.setPosition(x * cellSize, y * cellSize);

            // Cambia el color según el valor de la neurona (-1 o 1)
            if (pattern[index] == 1) {
                cell.setFillColor(sf::Color::Green);
            } else {
                cell.setFillColor(sf::Color::Red);
            }

            cell.setOutlineColor(sf::Color::Black);
            cell.setOutlineThickness(2);

            window.draw(cell);
        }
    }

    window.display(); // Muestra lo que se ha dibujado en la ventana
}

//Comenzando de condiciones iniciales al azar para las neuronas analice la evolución de la energía en función de los pasos de evolución de la red.
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

    int N = 1000;
    Red red(N);
    red.inicializarNeuronas(neuronasAlAzar);
    patron p;
    for(int i = 0; i < N; i++){
        p.push_back(2*(rand()%2)-1);
    }
    red.cargarPatron(p);
    red.entrenar();
    red.calcularUmbrales(umbralesNulos);

    float E;
    ofstream archivo;
    archivo.open("energia.txt");
    for(int i = 0; i < 10000; i++){
        red.evolucionar();
        E = red.obtenerEnergia();
        archivo << E << endl;
    }
}

//2. Partiendo de uno de los patrones a almacenar con algún número de modificaciones aleatorias, analice la convergencia de la red al patrón original, calculando el número de neuronas que difieren del patron original.
patron patronX = {
    1,-1,-1,-1,1,
    -1,1,-1,1,-1,
    -1,-1,1,-1,-1,
    -1,1,-1,1,-1,
    1,-1,-1,-1,1
};
patron patronCuadrado = {
    1,1,1,1,1,
    1,-1,-1,-1,1,
    1,-1,-1,-1,1,
    1,-1,-1,-1,1,
    1,1,1,1,1
};
patron patronAAlmacenar;
void neuronasConError(vector<Neurona>& neuronas){
    for(int i = 0; i < neuronas.size(); i++){
        neuronas[i] = patronAAlmacenar[i];
        if(rand()%100 < 10) neuronas[i] *= -1;
    }
}
const std::string imagePath = "train-images.idx3-ubyte";
int numImages, numRows, numCols;
void ej2(){
    srand(time(NULL));

    auto images = readMNISTImages(imagePath, numImages, numRows, numCols);
    patronAAlmacenar = binarizeImage(images[0]);

    int N = numRows*numCols;
    Red red(N);
    red.inicializarNeuronas(neuronasConError);
    red.cargarPatron(patronAAlmacenar);
    red.entrenar();
    red.calcularUmbrales(umbralesNulos);

    sf::RenderWindow window(sf::VideoMode(numRows * cellSize, numCols * cellSize), "Red de Hopfield");
    displayNetwork(window,patronAAlmacenar);
    sf::sleep(sf::seconds(1));

    ofstream archivo;
    archivo.open("errores.txt");
    for(int i = 0; i < 10000; i++){
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        red.evolucionar();
        vector<Neurona> neuronas = red.obtenerNeuronas();
        displayNetwork(window,neuronas);
        int errores = 0;
        for(int i = 0; i < N; i++){
            if(neuronas[i] != patronAAlmacenar[i]) errores++;
        }
        archivo << errores << endl;
        sf::sleep(sf::milliseconds(10));
    }
}

//3.Analice la capacidad de almacenamiento de la red. Para ello estudie el éxito al recuperar una memoria en función del número M de patrones almacenados. 
//Tenga en cuenta que el estado al que converge la red puede no diferir de los patrones de entrenamiento.
vector<patron> patronesAAlmacenar;
void ej3(){
    srand(time(NULL));

    auto images = readMNISTImages(imagePath, numImages, numRows, numCols);
    for(int i = 0; i < 6; i++){ //aca tengo dudas porque supongo que depende de que tan parecido sean los patrones
        patronesAAlmacenar.push_back(binarizeImage(images[i]));
    }
    //supongo que la memoria es = patronesAAlmacenar[0] y lo guardo en patronAAlmacenar
    patron memoria = patronesAAlmacenar[0];
    patronAAlmacenar = memoria;

    int N = numRows*numCols;
    Red red(N);
    red.inicializarNeuronas(neuronasConError);
    int M = patronesAAlmacenar.size();
    for(int i = 0; i < M; i++){
        red.cargarPatron(patronesAAlmacenar[i]);
    }
    red.entrenar();
    red.calcularUmbrales(umbralesNulos);

    ofstream archivo;
    archivo.open("exitos_M" + to_string(M) + ".txt"); //PUEDO PROBAR hacer producto punto entre neuronas y memoria para cuantificar el exito
    for(int i = 0; i < 5000; i++){
        red.evolucionar();
        vector<Neurona> neuronas = red.obtenerNeuronas();
        int errores = 0;
        for(int i = 0; i < N; i++){
            if(neuronas[i] != memoria[i]) errores++;
        }
        archivo << errores << endl;
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
}
void ej4(){
    srand(time(NULL));

    auto images = readMNISTImages(imagePath, numImages, numRows, numCols);
    patronAAlmacenar = binarizeImage(images[0]);
    patronesAAlmacenar.push_back(patronAAlmacenar);

    int N = numRows*numCols;

    //repito ej1
    Red red(N);
    red.inicializarNeuronas(neuronasAlAzar);
    red.cargarPatron(patronAAlmacenar);
    red.entrenar();
    red.calcularUmbrales(umbralesPromedio);

    float E;
    ofstream archivo;
    archivo.open("Resultados/umbrales_promedio/energia.txt");
    for(int i = 0; i < 5000; i++){
        red.evolucionar();
        E = red.obtenerEnergia();
        archivo << E << endl;
    }
    archivo.close();

    //repito ej2
    Red red2(N);
    red2.inicializarNeuronas(neuronasConError);
    red2.cargarPatron(patronAAlmacenar);
    red2.entrenar();
    red2.calcularUmbrales(umbralesPromedio);

    archivo.open("Resultados/umbrales_promedio/errores.txt");
    for(int i = 0; i < 5000; i++){
        red2.evolucionar();
        vector<Neurona> neuronas = red2.obtenerNeuronas();
        int errores = 0;
        for(int i = 0; i < N; i++){
            if(neuronas[i] != patronAAlmacenar[i]) errores++;
        }
        archivo << errores << endl;
    }
    archivo.close();

    //repito ej3
    for(int M = 1; M < 10; M++){
        patronesAAlmacenar.clear();
        for(int i = 0; i < M; i++){
            patronesAAlmacenar.push_back(binarizeImage(images[i]));
        }
        patron memoria = patronesAAlmacenar[0];
        patronAAlmacenar = memoria;

        Red red3(N);
        red3.inicializarNeuronas(neuronasConError);
        for(int i = 0; i < M; i++){
            red3.cargarPatron(patronesAAlmacenar[i]);
        }
        red3.entrenar();
        red3.calcularUmbrales(umbralesPromedio);

        archivo.open("Resultados/umbrales_promedio/exitos_M" + to_string(M) + ".txt");
        for(int i = 0; i < 5000; i++){
            red3.evolucionar();
            vector<Neurona> neuronas = red3.obtenerNeuronas();
            int errores = 0;
            for(int i = 0; i < N; i++){
                if(neuronas[i] != memoria[i]) errores++;
            }
            archivo << errores << endl;    
        }
        archivo.close();
    }
}

int main(){
    ej4();
    return 0;
}