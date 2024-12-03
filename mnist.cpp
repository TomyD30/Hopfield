#include "headers/mnist.hpp"

img cargarImagenesMNIST(const string& archivo, int& numImagenes, int& numFilas, int& numCols){
    ifstream file(archivo, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("No se pudo abrir el archivo de imágenes MNIST.");
    }

    int magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    magicNumber = __builtin_bswap32(magicNumber);
    if (magicNumber != 2051) {
        throw runtime_error("Número mágico inválido en el archivo de imágenes MNIST.");
    }

    file.read(reinterpret_cast<char*>(&numImagenes), 4);
    numImagenes = __builtin_bswap32(numImagenes);

    file.read(reinterpret_cast<char*>(&numFilas), 4);
    numFilas = __builtin_bswap32(numFilas);

    file.read(reinterpret_cast<char*>(&numCols), 4);
    numCols = __builtin_bswap32(numCols);

    img images(numImagenes, vector<uint8_t>(numFilas * numCols));
    for (int i = 0; i < numImagenes; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), numFilas * numCols);
    }

    file.close();
    return images;
}

imgB binarizarImagen(const vector<uint8_t>& imagen){
    imgB imagenBinaria(imagen.size());
    for (size_t i = 0; i < imagen.size(); ++i) {
        imagenBinaria[i] = (imagen[i] > 128) ? 1 : -1;
    }
    return imagenBinaria;
}

//corto una imagen y la convierte en L*L centrada
imgB cortarImagen(imgB& imagen, int L, int numRows, int numCols){
    imgB imagenCortada;
    imagenCortada.reserve(L*L);
    for(int i = (numRows-L)/2; i < (numRows+L)/2; i++){
        for(int j = (numCols-L)/2; j < (numCols+L)/2; j++){
            imagenCortada.push_back(imagen[i*numCols+j]);
        }
    }
    return imagenCortada;
}

void editarImagen(imgB& imagen, int numRows, int numCols){
    //agrego un borde con 1's a la imagen
    int ancho = 2;
    for(int i = 0; i < numRows; i++){
        for(int j = 0; j < numCols; j++){
            if(i < ancho || i >= numRows-ancho || j < ancho || j >= numCols-ancho){
                imagen[i*numCols+j] = 1;
            }
        }
    }
}
void editarImagen2(imgB& imagen, int numRows, int numCols){
    //agrego un borde aleatorio a la imagen
    int ancho = 14;
    for(int i = 0; i < numRows; i++){
        for(int j = 0; j < numCols; j++){
            if(i < ancho || i >= numRows-ancho || j < ancho || j >= numCols-ancho){
                imagen[i*numCols+j] = 2*(rand()%2)-1;
            }
        }
    }
}
void editarImagen3(imgB& imagen, int numRows, int numCols){
    //muevo el patron a un lugar aleatorio
    imgB aux = imagen;
    int ancho = 20;
    int r = rand()%4;
    int x,y;
    if(r == 0) x = -4, y = 4;
    else if(r == 1) x = 4, y = 4;
    else if(r == 2) x = -4, y = -4;
    else x = 4, y = -4;
    for(int i = 0; i < numRows; i++){
        for(int j = 0; j < numCols; j++){
            if(i >= x && i < x+numRows && j >= y && j < y+numCols){
                imagen[i*numCols+j] = aux[(i-x)*numCols+(j-y)];
            }
            else{
                imagen[i*numCols+j] = -1;
            }
        }
    }
}
