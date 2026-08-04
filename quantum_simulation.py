import torch

def ground_states():
    x = torch.tensor([1,0], dtype=torch.complex32)
    return x

def excited_states():
    x = torch.tensor([0,1], dtype=torch.complex32)
    return x

def H_0():
    return "I am going to go to bed now and work on this later, do not forget this Ammar"

def H_c():
    return "sleeeep now"