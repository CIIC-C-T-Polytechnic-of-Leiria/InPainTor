import torch

from model import InpainTor  # Import your model

# Set up debugging
torch.autograd.set_detect_anomaly(True)

# Initialize the model
model = InpainTor(selected_classes=[0], base_chs=16)
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor
# Adjust the dimensions based on your expected input size
batch_size = 1
channels = 3
height = 512
width = 512
dummy_input = torch.randn(batch_size, channels, height, width)


# Define a hook to print tensor shapes
def print_shape_hook(module, input, output):
    print(f"\n{module.__class__.__name__} forward pass:")
    print(f"  Input type: {type(input)}")
    if isinstance(input, tuple):
        for i, inp in enumerate(input):
            if isinstance(inp, torch.Tensor):
                print(f"  Input[{i}] shape: {inp.shape}")
            else:
                print(f"  Input[{i}] type: {type(inp)}")
    elif isinstance(input, torch.Tensor):
        print(f"  Input shape: {input.shape}")
    else:
        print(f"  Input type: {type(input)}")

    print(f"  Output type: {type(output)}")
    if isinstance(output, dict):
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  Output['{k}']: shape {v.shape}")
            else:
                print(f"  Output['{k}']: type {type(v)}")
    elif isinstance(output, torch.Tensor):
        print(f"  Output shape: {output.shape}")
    elif isinstance(output, tuple):
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                print(f"  Output[{i}] shape: {out.shape}")
            else:
                print(f"  Output[{i}] type: {type(out)}")
    else:
        print(f"  Output type: {type(output)}")


# Register hooks
model.shared_encoder.register_forward_hook(print_shape_hook)
model.segment_decoder.register_forward_hook(print_shape_hook)
model.generative_decoder.register_forward_hook(print_shape_hook)

# Run the model with dummy input
try:
    with torch.no_grad():
        output = model(dummy_input)
    print("\nFinal output shapes:")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")
except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback

    traceback.print_exc()
