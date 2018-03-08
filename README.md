# MNIST Train model

The dataset we need is the mnist dataset.

All the data can be found in [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

You need to download four files from mnist dataset website, and put them in MNIST package.
The MNIST package will be like this.

+ MNIST
    - t10k-images.idx3-ubyte
    - train-labels.idx1-ubyte
    - train-images.idx3-ubyte
    - t10k-labels.idx1-ubyte

With all the dataset are put in right dir, you can open your jupyter and run the Runnable.ipynb.

The function pybytes_to_int32 was introduce in mnist_helper.c. You need to compile it before run the code.

If your os is linux, run "gcc -fPIC -shared mnist_helper.c -o mnist_helper.so" in the terminal.

> Extra: Some of the code came from CS231n