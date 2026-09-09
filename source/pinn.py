import torch 
import quantum_simulation



class PINN(torch.nn.Module):
    
    def __init__(self, hidden_size = 64):
        super().__init__()

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

    def hamiltonian (self, t):
        
        # Depends on Hamiltonian ... What Hamiltonian do I use?
        # I could make this Hamiltonian an input variable so that this algorithm is more flexible
        pass

    def physics_residual(self, t):

        # Calculate the derivative of the wave function plus i H(t) psi(t) = 0

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