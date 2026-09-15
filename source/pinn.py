import torch 
import torch.nn as nn
import quantum_simulation

class PINN(torch.nn.Module):
    
    def __init__(self, hidden_size = 64, w_x=1.0, w_z=1.0):
        super().__init__()

        self.w_x = w_x
        self.w_z = w_z

        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 4)
        )
    def forward (self, t):
        return self.network(t)

    def wavefunction (self, t):
        # Convert four real inputs into complex wave function
        output = self.forward(t)

        re_c0 = output[:, 0]
        im_c0 = output[:, 1]

        re_c1 = output[:, 2]
        im_c1 = output[:, 3]

        c0 = torch.complex(re_c0, im_c0)
        c1 = torch.complex(re_c1, im_c1)

        return torch.stack([c0, c1], dim=1)

    def ctrl(t):
        return torch.tanh(t) 

    def hamiltonian (self, ctrl_fn = None):
        sigma_x = torch.tensor([[0,1],[1,0]],dtype=torch.complex64)
        sigma_z = torch.tensor([[1,0],[0,-1]],dtype=torch.complex64)
        sigma_ee = torch.tensor([[1,0],[0,0]],dtype=torch.complex64)
        if ctrl_fn is None:
            ctrl = torch.zeros_like(t)
        else:
            ctrl = ctrl_fn(t)
        H_t = (w_z*sigma_z + w_x*sigma_x + ctrl[:,None, None]*sigma_ee)
        return H_t

    def physics_residual(self, t):

        # Calculate the derivative of the wave function plus i H(t) psi(t) = 0
        Schrodinger_LHS = torch.autograd.grad(wavefunction(t))
        Schrodinger_RHS = torch.prod(hamiltonian(t),quantum_simulation.schrodinger_hamiltonian(psi, t))
        pass

    def physics_loss (self, t):

        # Calculates MSE of result
        residual = self.physics_residual(t)
        
        error = torch.mean(torch.abs(residual) ** 2)
        return error

    def initial_loss (self, t0, psi0):
        
        # Ensure that at time t=0, the value of the wave function is 0 
        pred  = self.wavefunction(t0)
        error = torch.mean(torch.abs(pred - psi0)**2)
        return error

    def normalized_loss (self, t):
        
        # Ensure that the area under the probability curve is 1 (necessary because probability)
        psi = self.wavefunction(t)
        norm = torch.sum(torch.abs(psi)**2, dim=1)
        error = torch.mean(torch.abs(norm-1.0)**2)
        return error        