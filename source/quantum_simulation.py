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

def time_evolution(T=10.0, dt=0.1,psi_0):
    psi = psi_0.clone()
    states = [psi.clone()]
    times=[0.0]

    while (t<T):
        slope_1 = schrodinger_hamiltonian(torch.tensor(t),psi) 
        #must make t a tensor so it has torch optimizations with control function
        bad_prediction = psi+dt*slope_1
        slope_2 = schrodinger_hamiltonian(torch.tensor(t+dt),bad_prediction)
        psi += (slope_1+slope_2)*0.5*dt

        states.append(psi.clone())
        times.append(t+dt) #Creating a giant table of psi for all values of t

        t += dt
    return torch.tensor(times), torch.stack(states)

def probability(psi):
    x = torch.abs(psi)**2 # Multiply by complex conjugate to get probability for each value of t
    return x

def norm(psi):
    x = torch.sum(probability(psi)) 
    return x #Is this inefficient for ram?
    #Find the sum of all proabilities for each value of t so we can normalize it and all probabilities add up to 1
