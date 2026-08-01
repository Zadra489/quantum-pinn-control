# Physics-Informed Neural Networks for Quantum Control

## Project Goal:

This project aims to reproduce and analyse the results seen in *Physics-informed neural networks for quantum control* by Norambuena et Al

https://arxiv.org/pdf/2206.06287

## Research Question:

How accurately can a PINN learn the evolution of a quantum system governed by the Schrodinger equation?

# Necessary Backround

## Quantum Dynamics

This is a field that I had little experience in and thus I have been teaching myself, using *Quantum Mechanics Made Simple: Lecture Notes* by Weng Cho 

https://wcchew.ece.illinois.edu/chew/course/QMALL20120717.pdf

While I did not have direct experience with quantum mechanics, I am taking a quantum course next year so I possess the relevant knowledge for an intro to quantum mechanics. I already understands mathematical concepts such as basic probability, statistics, complex analysis, and linear Algebra from courses and personal projects

I have been reading the textbook and building a comprehensive understanding of the topics needed for this projects, specifically the time-dependant Schrodinger equation and the Hamiltonian

Quantum Control Resources:
*Introduction to quantum control: From basic concepts to applications in quantum technologies* by Christiane P. Koch
*Quantum Optimal Control Theory* by J. Werschnik and E.K.U. Gross

## Physics-Informed Neural Networks:

While a normal neural networks learn from data and utilize the difference between the predicted value and expected value in the loss function, physics-informed neural networks utilize laws of physics which we know the model must uphold to be accurate. The advantage is that we can create more complex loss functions that into account various factors such as:

1. Physics loss: how closely the solution satisfies the Schrodinger equation
2. Boundry loss: how closely the solution follows the initial quantum state
3. Target-state loss: how closely the solution matches the desired outcome

# Completed:
- Read paper on which this project is based on
- Setup Python enviroment
- Created GitHub repo
- Configured VSC
- Installed all required Python Packages

# To Do:

## Short Term:

- Implement quantum two-level system simulator
- Implement Hamiltonian definition and Schrodinger equation solver
- Implement State evolution visualization

# Long Term:

- Neural network architecture
- Automatic differentiation
- Physics-based loss function
- Training procedure
- Compre PINN results and Numerical solver results

Measure fidelity, convergence, control accuracy

# Enviroment:

I am dualbooting between Windows 11 and Arch Linux 

Development via Visual Studio Code on Windows 11 and Visual Studio Codium on Arch Linux

Libraries:
- PyTorch (for neural network, automatic differentiation)
- NumPy (for numerical solver)
- Scipy (to implement hamiltonian and schrodinger equation)
- Matplotlib (to visualize results)
- Jupyter (to record results and notes) 

