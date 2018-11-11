# deeplearning_adam
ADAM on MNIST data for Deep Learning coursework 1

Run get_data.py first to read the data in (ensure it is in the same local directory).
Then run adam.py in Kernel namespace.

At the moment the code just does gradient descent (not SGD) with a constant update 
x_{k+1} = x_k - \epsilon * g_k
but the weights learned are not useful.

To change to SGD and using Adam un-comment the respective lines in the code.
