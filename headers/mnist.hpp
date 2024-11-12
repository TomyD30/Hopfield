#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

using namespace std;

vector<vector<uint8_t>> readMNISTImages(const string& filePath, int& numImages, int& numRows, int& numCols);
vector<int> binarizeImage(const vector<uint8_t>& image);
