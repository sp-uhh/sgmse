import torch


def device_agnostic_enhance(model, input_tensor):
    """
    Enhance the input tensor using the model in a device-agnostic manner.
    Works with CPU, CUDA, and MPS (Apple Silicon).
    """
    # Determine the device
    device = input_tensor.device

    # Move model to the same device as input tensor
    model.to(device)

    # Perform enhancement
    with torch.no_grad():
        enhanced_output = model(input_tensor)
    return enhanced_output


if __name__ == '__main__':
    # Example usage
    # Assuming 'my_model' is your model and 'input_data' is your input tensor
    my_model = ...  # Load or define your model here
    input_data = ...  # Load or define your input tensor here
    output = device_agnostic_enhance(my_model, input_data)
    print(output)