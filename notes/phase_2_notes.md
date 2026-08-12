# Plan:

Now that I have a mathematical reference to calculate the wave function, I can begin creating a neural network which also attempts to create the wave function. 

However, an important design choice is how I represent the quantum state. The quantum state is represented by a vector valued function where each component represents the amplitude of each state (0 and 1). Since both of these amplitudes are complex I will represent the real and imaginary components for the amplitudes of each state. 

Therefore, the neural network will take in a single input of t and output the real and imaginary components for two wave functions. 

Once I have estimated the wave function at a specific value of t, I will use pytorch autograd to estimate the derivative of psi with respect to t. 

With this, I can create a loss function which can use schrodinger's equation to estimate how inaccurate the derivative of the wave function is via the residual. I can also create a loss condition based on how much it satisfies the initial condition and how normalized it is. 

Overall, I want to see if a neural network can learn what a quantum system does while obeying Schrodinger's equation. I will later try to use this to optimize u(t) to control the quantym system. 

# PINN Design:

