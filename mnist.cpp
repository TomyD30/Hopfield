#include "headers/mnist.hpp"

// Lee el encabezado de las imágenes MNIST y devuelve un vector de imágenes
vector<vector<uint8_t>> readMNISTImages(const string& filePath, int& numImages, int& numRows, int& numCols) {
    ifstream file(filePath, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("No se pudo abrir el archivo de imágenes MNIST.");
    }

    int magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    magicNumber = __builtin_bswap32(magicNumber);
    if (magicNumber != 2051) {
        throw runtime_error("Número mágico inválido en el archivo de imágenes MNIST.");
    }

    file.read(reinterpret_cast<char*>(&numImages), 4);
    numImages = __builtin_bswap32(numImages);

    file.read(reinterpret_cast<char*>(&numRows), 4);
    numRows = __builtin_bswap32(numRows);

    file.read(reinterpret_cast<char*>(&numCols), 4);
    numCols = __builtin_bswap32(numCols);

    vector<vector<uint8_t>> images(numImages, vector<uint8_t>(numRows * numCols));
    for (int i = 0; i < numImages; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), numRows * numCols);
    }

    file.close();
    return images;
}

vector<int> binarizeImage(const vector<uint8_t>& image) {
    vector<int> binaryImage(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        binaryImage[i] = (image[i] > 128) ? 1 : -1;
    }
    return binaryImage;
}
