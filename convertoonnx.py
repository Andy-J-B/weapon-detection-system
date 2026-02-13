import torch
import torch.nn as nn


# 1. Define or Load your model architecture
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()

# 2. Load your weights (if you have a .pt file)
model.load_state_dict(
    torch.load(
        "/Users/Andy_1/dev/code/programs/GitHub/weapon-detection-system/runs/detect/runs/train/weapon_binary/weights/best.pt",
        map_location="cpu",
    )
)

# 3. SET TO EVALUATION MODE - Crucial for dropout/batchnorm layers
model.eval()

# 4. Create a dummy input matching your model's input shape
# (Batch Size, Input Features)
dummy_input = torch.randn(1, 10)

# 5. Export the model
torch.onnx.export(
    model,  # The model to be exported
    dummy_input,  # A sample input tensor
    "model.onnx",  # The output filename
    export_params=True,  # Store the trained weights inside the file
    opset_version=11,  # The ONNX version to use
    do_constant_folding=True,  # Optimize the graph
    input_names=["input"],  # Optional: name for the input node
    output_names=["output"],  # Optional: name for the output node
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },  # Optional: allow variable batch sizes
)

print("Model successfully converted to model.onnx")
