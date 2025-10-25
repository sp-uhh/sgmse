import torch

def enhance(model, data):
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Move model to the appropriate device
    model.to(device)
    
    # Move data to the appropriate device
    data = data.to(device)

    # Perform the enhancement
    output = model(data)
    
    return output
