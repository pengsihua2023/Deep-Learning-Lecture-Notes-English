# Mathematical Description

A **fully connected network** (dense network) consists of multiple layers of neurons.  
Each neuron in one layer is connected to **all** neurons in the next layer.



### (1) Input

$$
\mathbf{x} \in \mathbb{R}^{d}
$$

*The input vector with dimension $d$.*



### (2) Linear Transformation

$$
\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, 
\quad \mathbf{a}^{(0)} = \mathbf{x}
$$

*Computes the pre-activation for layer $l$.*



### (3) Activation

$$
\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})
$$

*Applies the activation function elementwise.*



### (4) Output

$$
\mathbf{y} = \mathbf{a}^{(L)}
$$

*Final output of the network.*  

- For **regression**: $\sigma$ may be the identity function.  
- For **binary classification**: $\sigma$ is often sigmoid.  
- For **multi-class classification**: $\sigma$ is typically softmax.  

