import torch

def ground_states():
    x = torch.tensor([1,0], dtype=torch.complex32) #Complex32 for simplicity
    return x

def excited_states():
    x = torch.tensor([0,1], dtype=torch.complex32)
    return x

def H_0(energy_gap = 1.0): #Use energy gap instead of unique energies to simplify calculations
    x = torch.tensor([
            [0.0, 0.0],
            [0.0, energy_gap]],
        dtype=torch.complex32)
    return x

def H_c():
    x = strength * torch.tensor([ #Don't need input variable since u(t) controls it's amplitude anyways
        [0.0, 1.0],
        [1.0, 0.0]],
        dtype=torch.complex64
    )
    return x

def u(t, A  = 5.0, d = 2.0, k = 1.0):
    x = A * torch.exp( -((t - center) ** 2) / (width ** 2))
    return x

    # Here, A is vertical stretch, d is horizontal shift, and k is horizontal compression.
    # I chose a gaussian curve as it is the shape that minimizes the uncertainty between the momentum and position as seen in the heisenberg uncertainty principles
    # I forgot which article on physicslibre texts I read that from but it was somewhere

def H(u, t):
    return H0() + u(t) * Hc() # the control function is the only one dependant on time

def schrodinger_hamiltonian(t, psi): # schrodinger equation that uses hamiltonian and psi
    return -1j * H(t) @ psi # using psi for torch matric multiplication

