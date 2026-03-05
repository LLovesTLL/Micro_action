
import torch
import sys
import os

def check_ckpt(path):
    print(f"Checking checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Found 'model' key in checkpoint.")
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']
        print("Found 'module' key in checkpoint.")
    else:
        state_dict = checkpoint
        print("Using checkpoint root as state_dict.")

    print(f"Total keys: {len(state_dict)}")
    
    # Check patch_embed keys
    patch_keys = [k for k in state_dict.keys() if 'patch_embed' in k]
    print("\nPatch Embed Keys:")
    for k in patch_keys:
        print(f"  {k}: {state_dict[k].shape}")

    # Check first few keys
    print("\nFirst 10 keys:")
    for k in list(state_dict.keys())[:10]:
        print(f"  {k}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default path
        path = "videomambapro/pretrained_models/videomambapro_t16_in1k_res224.pth"
    
    if os.path.exists(path):
        check_ckpt(path)
    else:
        print(f"File not found: {path}")
