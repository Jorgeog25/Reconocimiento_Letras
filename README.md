# OCR con Red Neuronal Convolucional (CNN)

+ [Documento](./docs/OCR%20con%20Red%20Neuronal%20Convolucional.pdf)
+ [Video](https://drive.google.com/file/d/1_hKUdcG9vlW9OvcEe6_rIXbeknfx-lTL/view?usp=drive_link)

## Código y entorno de ejecución

El proyecto se ha desarrollado íntegramente en **Python**, utilizando archivos ejecutables (`.py`) organizados en un entorno de desarrollo estándar (IDE).  
El sistema no requiere el uso de notebooks Jupyter ni entornos externos para su ejecución, aunque el código podría adaptarse fácilmente a dicho formato si fuese necesario.

El OCR implementado se ejecuta como una aplicación local, recibiendo como entrada imágenes en formato estándar y produciendo como salida el texto reconocido, junto con una visualización opcional del proceso de segmentación y clasificación.

---

## Dependencias y librerías utilizadas

Para la ejecución del proyecto es necesario disponer de las siguientes librerías externas, todas ellas de uso general y **sin funcionalidades OCR integradas**, cumpliendo así las restricciones del trabajo.

### Python
- **Versión recomendada:** Python 3.9 o superior  
- **Fuente:** https://www.python.org  
- **Comprobación de versión:**
```bash
python --version
```

---

### OpenCV
- **Versión utilizada:** 4.x  
- **Fuente:** https://opencv.org  

#### Uso en el proyecto
OpenCV se utiliza como base para todo el procesamiento de imagen previo al reconocimiento, incluyendo:

- Conversión de imágenes a escala de grises  
- Binarización adaptativa de imágenes  
- Operaciones morfológicas para limpieza del ruido  
- Segmentación de caracteres mediante análisis de componentes conexas  

El uso de OpenCV se limita estrictamente a tareas de **visión artificial clásica**, sin emplear en ningún caso funcionalidades OCR integradas, cumpliendo así las restricciones del proyecto.

#### Instalación
```bash
pip install opencv-python
```

---

### TensorFlow / Keras
- **Versión utilizada:** TensorFlow 2.x (incluye Keras)  
- **Fuente:** https://www.tensorflow.org  

#### Uso en el proyecto
- Definición de la Red Neuronal Convolucional (CNN)  
- Entrenamiento desde cero del modelo  
- Inferencia del clasificador de caracteres  

#### Instalación
```bash
pip install tensorflow
```

---

### NumPy
- **Versión utilizada:** 1.x  
- **Fuente:** https://numpy.org  

#### Uso en el proyecto
- Manipulación de matrices  
- Cálculo de métricas y descriptores visuales  

#### Instalación
```bash
pip install numpy
```

---

### TensorFlow Datasets (opcional, para entrenamiento)
- **Versión utilizada:** compatible con TensorFlow 2.x  
- **Fuente:** https://www.tensorflow.org/datasets  

#### Uso en el proyecto
- Carga del dataset EMNIST para ampliar el conjunto de entrenamiento manuscrito  

#### Instalación
```bash
pip install tensorflow-datasets
```

---

## Notas importantes

- No se utiliza ninguna librería OCR externa ni soluciones de reconocimiento de texto preentrenadas (como Tesseract, EasyOCR o APIs comerciales).
- Todas las funcionalidades de segmentación, normalización y clasificación han sido implementadas explícitamente como parte del proyecto.
- El sistema ha sido diseñado para ser ejecutado de forma local y reproducible en cualquier entorno compatible con las versiones indicadas.

---

## Ejecución del sistema

Para ejecutar el OCR sobre una imagen:

```bash
python -m src.main ruta/a/la/imagen.png
```

El sistema mostrará opcionalmente una ventana de depuración con la segmentación de caracteres y el resultado del reconocimiento, y devolverá por consola el texto reconocido.
