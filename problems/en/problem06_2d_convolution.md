# Problem 6: 2D Convolution

2D convolution is a fundamental operation in digital image processing, computer vision, and deep learning. The aim of this problem is to implement 2D convolution through programming, further consolidating your understanding and application of parallel computing, thread cooperation, and shared memory.

## 2D Convolution

2D convolution is one of the most basic and important operations in digital image processing. Essentially, it uses a small matrix known as a convolution kernel or filter, to compute a weighted sum over each pixel and its neighborhood in the image. This operation enables effects like image smoothing, sharpening, and edge detection, and serves as the foundation for many advanced image processing algorithms.

From a mathematical perspective, the 2D discrete convolution operation can be expressed as follows: for an input image $I$ and a convolution kernel $K$ of size $m \times n$, the pixel value at position $(x, y)$ in the output image $I'(x, y)$ is the sum of the products between the corresponding elements of the image region centered at $(x, y)$ and the kernel $K$. The precise formula is:

$$I'(x, y) = \sum_{i=0}^{m-1}\sum_{j=0}^{n-1} I(x+i, y+j) \cdot K(i, j)$$

Here, the ranges of $i$ and $j$ are determined by the kernel size. Intuitively, this process involves sliding the convolution kernel over the image and performing a local weighted summation at each position.

```
Input:            Kernel:      Output:
┌─────────┐     ┌──────┐     ┌───────┐
│ 1  2  3 │     │ 1  1 │     │ 12 15 │
│ 4  5  6 │  *  │ 1  1 │  =  │ 24 28 │
│ 7  8  9 │     └──────┘     └───────┘
└─────────┘
```

If no special handling is done at the boundaries, the output image will be smaller than the input image (specifically, output size = input size - kernel size + 1). To keep the output image the same size as the input, it's common to add extra border pixels to the input image—a process known as padding. Padding can be done with zeros (zero-padding) or by mirroring (using values near the edge). In this problem, you are required to use zero-padding to handle the boundaries.

Tip: As with Problem 3, you can use shared memory to optimize your computation.

## Testing

In the test data, the length and width of the convolution kernel may be any of $\{1, 3, 5, 7\}$. You can still use `make test` to compile and run the tests.