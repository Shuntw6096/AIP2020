# Advanced Image Processing
NTNU-CS Advanced Image Processing 2020 Fall, image processing software implementation.

### Function：
1. Read image files
2. Image histogram
3. Generation of additive, zero mean Gaussian noise
4. Discrete wavelet transform
5. Histogram equalization
6. Image smoothing and edge detection

### Requirements
Detail in Pipfile
1. numpy
2. matplotlib
3. tkinter-python3

#### Function：Read image files
The function can read and write image files with different format, including JPG, BMP, or PPM.  
![func1](https://github.com/Shuntw6096/AIP2020/blob/main/img/func1.jpg)

#### Function：Image histogram
The function can show the histogram of an image, left image is input, right image is output.  
![func2](https://github.com/Shuntw6096/AIP2020/blob/main/img/func2.jpg)

#### Function：Generation of additive, zero mean Gaussian noise
The function can generate additive, zero mean Gaussian noises, left image is noisy image, right image is the histogram of Gaussian noise, and let users determine the variation or standard deviation of Gaussian noise.  
![func3](https://github.com/Shuntw6096/AIP2020/blob/main/img/func3.jpg)

#### Function：Discrete wavelet transform
The function can process discrete wavelet transform, the mother wavelet is Haar wavelet, and let users determine the number of iterations, maximum of iterations will be determined by input image size.  
![func4](https://github.com/Shuntw6096/AIP2020/blob/main/img/func4.jpg)

#### Function：Histogram equalization
The function can do histogram equalization on gray-level image, if input image is RGB, then will covert to gray-level image and do histogram equalization.  
![func5](https://github.com/Shuntw6096/AIP2020/blob/main/img/func5.jpg)

#### Function：Image smoothing and edge detection
The function can do image smoothing by Gaussian filter at first, users can determine standard deviation of Gaussian filter, and do convolution with different size of mask(square or non-square).  
convolution coefficient string：
1. square： `1 2 3 4 5 6 7 8 9` and `1 1 1 0 0 0 -1 -1 -1`
2. non-square： `1 2 3 -1 -2 -3 (2,3)` and `1 1 1 -1 -1 -1 (3,2)`
3. known operator：sobel(1~8),  kirsch(1~8), prewitt(1~8), laplace(4,8), case insensitive, i.e., `laplace-4` or `sobel-7`.
  
![func6](https://github.com/Shuntw6096/AIP2020/blob/main/img/func6.jpg)

## Final Project：Face Detection with Segmentation and Support Vector Machine
In this project, I research in region of interest proposal method just like R-CNN, implementation
methods can be divided into the following steps:
1. Build Color Model
2. Skin-like pixel detection using evolutionary agents
3. Skin-like region segmentation and adjacent region graph
4. SVM Training

Detail in Finalproject.pdf and AIP2020_face_detection.ipynb
