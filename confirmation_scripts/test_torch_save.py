import torch
import tempfile
import os

print(f"torch version: {torch.__version__}")

obj = {"key": "value", "number": 42}
with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
    path = f.name

try:
    torch.save(obj, path)
    print("torch.save: OK")
except Exception as e:
    print(f"torch.save: FAILED — {e}")
finally:
    os.unlink(path)
