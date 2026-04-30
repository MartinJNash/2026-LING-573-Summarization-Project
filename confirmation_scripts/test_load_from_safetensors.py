from safetensors.torch import load_file
from safetensors import safe_open

ADAPTER_PATH = "models/bart_base_trained_v1/adapter_model.safetensors"
tensors = load_file(ADAPTER_PATH)

print(f"Loaded {len(tensors)} tensors")
for name, tensor in tensors.items():
  print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")
