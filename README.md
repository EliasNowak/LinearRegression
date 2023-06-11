# Linear Regression

### Situation 
 I have some data, and would like to build a model using this training data to predict the y values of future x values. 




### Procedure

- Model

$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$

- Parameters

w,b

- cost function

$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$

- goal:

minimize J(w,b)