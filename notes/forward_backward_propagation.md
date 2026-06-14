#### Concept 
- Forward propagation := passing inpyut data through layers and compute output (prediction)
- Backward propagation := gradient of sloss function w.r.t weight abnd bias
	- using chain rule : dL/dW = dL/dY * dY/dW
- weight update := W_new = W_old - learning_rate * \partial{L}/\partial{W} 
- activation layer (a) := application of non linear function to output of neuron, and fires to a degree
- ReLU (Rectified Linear Unit) function := returns max(0, Z), also positive values 
- ReLU derivative := step function catching inputs that are positive 

#### Workflow
Input -> Forward Propagation -> Loss calculation -> Backward Propagation -> Weight update


