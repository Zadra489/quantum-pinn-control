# Expirement Design:

## For my initial implementation of the PINN, I will simulate the simplest possible quantum system, a closed two-level quantum system using the Schrodinger Equation

I chose for the quantum system to be closed as it allows for me to study the effectiveness of the PINN for quantum control under ideal circumstance. While the original paper generated Gibbs states and implemented the Markovian master equation, I chose to implement a closed system instead to better isolate the effectiveness of the control algorithm without the complexity and interference of external factors. Additionally, I have found that modelling the system as being closed is standard for an introduction to quantum control and appropriate when the enviromental effects are negligible in small timeframes. 

I chose for the quantum system to be two-level (aka electrons can only move between their base energy level and the next lowest energy level) as it simulates the conditions used in the original paper and is very common in the field of quantum control. It removes additional complexity from my expirementationa and allows me to simulate a quantum system while ensuring I have a low computing overhead, allowing for greater efficiency when training models. 

## I will also implement the time-dependent Hamiltonian and the time-dependent Schrodinger Equation

I need to use the time-dependent Hamiltonian and time-dependent Schrodinger Equation as the control input is inherently a function of time and thus the wave function also alters as a function of time. Therefore, it is necessary I implement the time-dependent Hamiltonian and time-dependent Schrodinger Equation. 

# Implementation:

## Schrodinger Equation:

In quantum_simulator.py, I will have functions with define the two level quantum state (having a 2x1 vector that allows complex units), the natural Hamiltonian (H_0), the control Hamiltonian (H_c), the control pulse (u(t)), and the Schrodinger equation (I am not writing it down in a markdown file like this, you know what it looks like).

For my simple two-level implementation, I will only use the energy difference for the Hamiltonians as it simplifies debugging and simply uses a difference "frame of reference" (to add a classical analogy for what I am doing)

I will then combine all these things into one function which solves how the quantum system evolves over time and perform basic tests to ensure it runs as expected.

Thus, I will be implementing functions for ground states, excited states, H_0, H_c, u(t), schrodinger(t, psi), and quantum_evolution()

## Implementation of Complex Numbers

While I could implement my own implementation of imaginary numbers, I need to ensure all operations are *extremely* efficient due to how many times they will be performed while training. Thus, I will be using the pytorch implementation of vectors (via tensors) and their built in application of complex numbers as the pytorch library is optimized for efficient training. 

I will use torch.complex32 for my initial implementation. While it does lack precision, it is much better on memory (and I just learned I will be forced to run this project on a laptop with 8gb on ddr3 for the next little bit due to unforseen circumstance :'). For a more refined version of the function (once I know everything works) to be used in the PINN, I will update it to torch.complex64 for more precision and if my training times are reasonable, torch.complex128 for even more precision to reduce systematic error and to ensure that all discrepancy with my desired results is due to the neural network when creating the loss function. 

## Time-Evolution:

In order to create an effective loss function for the AI, I need a method for solving the hamiltonian side of the schrodinger equation ()