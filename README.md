# Red de Hopfield - FisCom

## Funcionamiento general de la red
Los archivos [hopfield.cpp](./hopfield.cpp) y [hopfield.hpp](./headers/hopfield.hpp) describen el funcionamiento general de la red.
Los archivos [ejercicios.cpp](./ejercicios.cpp) y [ejercicios.hpp](./headers/ejercicios.hpp) tienen los análisis que se hicieron sobre la red.
Se requiere de la librería [OpenMP](https://www.openmp.org/) para la pararelización del código.
## Imagenes utilizadas para entrenar la red
Los archivos [mnist.cpp](./mnist.cpp) y [mnist.hpp](./headers/mnist.hpp) tienen funciones utilizadas para el manejo de las imagenes de MNIST utilizadas para entrenar la red.
Los patrones utilizados se encuentran en el archivo [train-images.idx3-ubyte](./train-images.idx3-ubyte).
## Graficos de la red
Los archivos [sfml.cpp](./sfml.cpp) y [sfml.hpp](./sfml.hpp) tienen funciones definidas para graficar en pantalla la red si se quiere.
Se requiere descargar la librería [SMFL Graphics](https://www.sfml-dev.org/).

## Utilización del código
En el archivo [main.cpp](./main.cpp) se llama a la funcion que se quiera correr.
Para compilar el progama se requieren los parámetros -fopenmp (para la paralelización del código) y incluir las librerías de SFML si se requiere con los parámetros -I SFML-2.6.1/include -L SFML-2.6.1/lib -lsfml-graphics -lsfml-window -lsfml-system y los archivos .dll dentro de la carpeta.
Ejemplo:
```sh
g++ -fopenmp *.cpp -o main.exe -I .../SFML-2.6.1/include -L .../SFML-2.6.1/lib -lsfml-graphics -lsfml-window -lsfml-system


