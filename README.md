# BlekNN
A neural network engine I made for fun, entirely in python, with no external libraries, with the bulk of the code having been written in one night. Note that due to the time constraints, certain omptimizations and feautures that may be present in other neural network engines are still yet to be added. 

# Requirements/Limitatons
While as of now the entire project does not need any additional libraries or external dependencies, it has only been tested in python 3.6. Although I have included a few activation and error functions, and their respective derivative functions, these are not using libraries that may optimize the calculation of these values. For example the value of 'e' in the sigmoid function is an approximation. 

# Quick start
BlekNN uses SGD (stochastic-gradient descent) to learn. 
After importing nn.py into your python project, you have to create the actual network.
This is done using the `network( )` constructor.
<br/> ```net = nn.network([3,5,4,2],[nn.funcs.sigmoid,nn.funcs.dsigmoid],[nn.funcs.mserr,nn.funcs.dmserr]) ```  
<br/>
<br/> This will create a neural network with an input size of 3, an output size of 2, with 2 hidden layers of size 5 and 4.
<br/> The second input list to the function, contains the activation function in index 0, and the derivative of the activation function in index 1. 
<br/> The third input list contains the error function in index 0, and its corresponding derivative function in index 1.
<br/> Define all your inputs as a 2d list 
<br/>```inputs = [[20,2,3],[29,3,4]]``` 
<br/> This is basically 2 sets of inputs
<br/> In this exapmple your output list must also be defined in a similar way, but each subset must contain 2 values corresponding to the 2 outputs of the neural network.
<br/> In order to train the neural network you simply have to run the following line: ```net.run(inputs,outputs)```
<br/> In order to use a trained network run the following line: ```net.feed_fwd(input_array)```

# Customization
Apart from the features mentioned previously you can also configure the number of iterations for training and the step size for learning.
This can be done using the `configure_runtime( )` method.
<br/> Example:
<br/>```configure_runtime( iterations = NUM_ITR, step_size = STP_SZE) ```





