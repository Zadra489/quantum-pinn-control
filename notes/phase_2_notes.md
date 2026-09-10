# Plan:

Now that I have a mathematical reference to calculate the wave function, I can begin creating a neural network which also attempts to create the wave function. 

However, an important design choice is how I represent the quantum state. The quantum state is represented by a vector valued function where each component represents the amplitude of each state (0 and 1). Since both of these amplitudes are complex I will represent the real and imaginary components for the amplitudes of each state. 

Therefore, the neural network will take in a single input of t and output the real and imaginary components for two wave functions. 

Once I have estimated the wave function at a specific value of t, I will use pytorch autograd to estimate the derivative of psi with respect to t. 

With this, I can create a loss function which can use schrodinger's equation to estimate how inaccurate the derivative of the wave function is via the residual. I can also create a loss condition based on how much it satisfies the initial condition and how normalized it is. 

Overall, I want to see if a neural network can learn what a quantum system does while obeying Schrodinger's equation. I will later try to use this to optimize u(t) to control the quantym system. 

# PINN Design:

I am trying to create a system that attempts to learn | psi(t) > given t, where the column vector of the wave function is represented as [ c_0 (t) , c_1 (t) ] for some functions of t. 

However, both of these components of the wave function are complex numbers. Thus, in order to decrease the computational complexity of the program, I will reprecent it as a 2x2 matrix where each column is a row vector for the real and imaginary components of that point. 

## Loss Function

Given that we are trying to craft a loss function using the residual between the PINN output and the output of quantum_simulation.py, we can incorporate what we know from the Schrodinger equation to build out our residual. Since the PINN calculates one side of the Schrodinger equation and quantum_simulation.py calculates the other half, we can subtract them to calculate the error of the PINN. Since the wave function is many points, we can find the standard deviation between the PINN and quantum_simulation.py as the residual.

Additionally, we can use the boundry conditions to reinforce training for the PINN. Since the beginning and end states are known, we can apply the MSE between the actual and prediced boundry points to the PINN.

Lastly, we know that the norm of the wave function must be one (because that's how probability works) so thus we can use the MSE between the calculated norm of the wave function produced by the PINN and the intended answer (1) to further reinforce training to be accurate for the PINN. 

We then add all the losses to create a single loss function. We could remove any of the loss conditions or create coefficients between the loss terms ... and I don't know how that would affect the accuracy of the model so I shall try to test that out


## Physics Residual:

The main component of the loss function exists to ensure the solution matches the Schrodinger equation, using the calculations creating in quantum_simulation.py

I will use quantum_simulation.py to make pinn.py more effective. Other than that, all I have to implement is autograd to calculate the other side of the Schrodinger equation

## Hamiltonian:

The research paper I am basing this project around uses a weird Hamiltonian ... for now I think I will try to implement making the Hamiltonian as an input variable but if that is not possible then I will use a simple matrix (based off of bloch spheres I suppose) for the time being, ensuring my PINN works in the first place

Ok so after further research I found out that the Hamiltonian the paper uses is (bare with me as I use latex without a latex thing but you can probably read this anyways so it's ok):
H(t) = w_z \sigma_z + w_x \sigma_x + \zeta (t) \sigma_{ee} 

the sigma terms obviously refer to base matricies which have visual corelations via Bloch spheres. The coefficients are system parameters, and thus will be put as inputs to the Hamiltonian function. The final term is essential, as it encodes the unknown control field and prevents the Hamiltonian from being constant. 

\sigma_{ee} is new notation from this paper and represents " Quantum transitions can be written in terms of the operators σij = |i⟩⟨j| (i, j = e, g) being |e⟩ (|g⟩) the excited (ground) state."

