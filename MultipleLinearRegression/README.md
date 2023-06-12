# Multiple Linear Regression

### Situation 
I have some data from: https://datahub.io/machine-learning/iris/datapackage.json
Simplified it looks something like this:

| sepallength | sepalwidth | petallength | petalwidth | class       |
|-------------|------------|-------------|------------|-------------|
| 2           | 5.1        | 3.5         | 1.4        | Iris-setosa |
| 4.9         | 3.0        | 1.4         | 0.2        | Iris-setosa |
| 5.0         | 2.4        | 1.5         | 0.2        | Iris-setosa |

And my goal is to develop a model that with the parameters sepallength, sepalwidth and petallength  predicted the petalwidth


### Procedure

- Model

$$ f_{\mathbf{w},b}(\mathbf{x}) =  w_0x_0 + w_1x_1 +... + w_{n-1}x_{n-1} + b$$

- Parameters

w,b

- cost function

$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2$$ 

- goal:

minimize J(w,b)


### Gradient descent

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j}  \; & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
\end{align*}$$

$$
\begin{align}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}   \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) 
\end{align}
$$
* m is the number of training examples in the data set


