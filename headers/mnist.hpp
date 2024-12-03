#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

using namespace std;

using img = vector<vector<uint8_t>>;
using imgB = vector<int>;

// Lee el encabezado de las imágenes MNIST y devuelve un vector de imágenes
img cargarImagenesMNIST(const string& archivo, int& numImagenes, int& numFilas, int& numCols);
// Convierte la imagen a -1 y 1
imgB binarizarImagen(const vector<uint8_t>& imagen);
imgB cortarImagen(imgB& imagen, int L, int numRows, int numCols);
void editarImagen(imgB& imagen, int numRows, int numCols);
void editarImagen2(imgB& imagen, int numRows, int numCols);
void editarImagen3(imgB& imagen, int numRows, int numCols);
