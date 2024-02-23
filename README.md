# Canny Edge Detection with Python

This project is a Python application that performs edge detection on an image using the Canny edge detection algorithm.

## Installation

1. Clone the project:
   ```bash
   git clone https://github.com/username/canny-edge-detection.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Add the `edge_detection.png` file to the root directory of the project.

## Usage

```python
python main.py
```

After running this command, the edge detection process will be performed, and the results will be saved in a file named `result.png`.

## Algorithm

The Canny edge detection algorithm consists of the following steps:

1. **Gaussian Blur**: The image is convolved with a Gaussian filter to reduce noise. This step smooths the image and helps in suppressing noise and minor variations in intensity.
   
2. **Gradient Calculation**: Sobel operators are applied to the blurred image to compute the gradients in both the horizontal and vertical directions. These gradients represent the rate of change of intensity in the image.
   
3. **Gradient Magnitude and Direction**: The gradient magnitudes and directions are calculated from the horizontal and vertical gradients. The gradient magnitude represents the strength of the edges, while the gradient direction indicates the direction of the edges.
   
4. **Non-Maximum Suppression**: Non-maximum pixels in the gradient magnitude image are suppressed to obtain a thinned edge map. This step ensures that only local maxima in the gradient direction are preserved as edge pixels.
   
5. **Hysteresis Thresholding**: This step aims to connect edge pixels into continuous edge contours. It involves two threshold values: a high threshold and a low threshold. Pixels with gradient magnitudes above the high threshold are considered strong edge pixels, while those below the low threshold are considered non-edge pixels. Pixels with gradient magnitudes between the two thresholds are considered weak edge pixels. Weak edge pixels are considered part of an edge if they are connected to strong edge pixels. This process helps in reducing the number of spurious edges.

## Example

```python
import cv2
from canny_edge_detection import canny_edge_detection

image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
edges = canny_edge_detection(image, low_threshold=50, high_threshold=150, kernel_size=3)

cv2.imshow('Original Image', image)
cv2.imshow('Edge Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Contributions

- You can contribute to this project by opening a pull request.
- If you encounter any bugs or issues, please open an issue to let us know.

