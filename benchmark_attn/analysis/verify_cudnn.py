import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
import sys

def verify_cudnn():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("Error: CUDA not available.")
        sys.exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # Dummy tensors
    B, N, H, D = 2, 1024, 4, 64
    # F.scaled_dot_product_attention expects q, k, v to be (*, H, N, D)
    q = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')

    print("Attempting to run SDPA with CUDNN_ATTENTION only...")
    try:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v)
        print("SUCCESS! CUDNN_ATTENTION ran successfully.")
    except Exception as e:
        print(f"HARD CRASH: Failed to run CUDNN_ATTENTION.\nError details:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_cudnn()
