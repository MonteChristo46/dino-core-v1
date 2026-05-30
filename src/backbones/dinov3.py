import torch.nn as nn
import transformers
import os

class DinoV3Backbone(nn.Module):
    def __init__(self, weights_dir="weights/dinov3"):
        super().__init__()
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(f"DINOv3 weights not found at {weights_dir}. Please run scripts/download_backbone.py first.")
            
        self.model = transformers.AutoModel.from_pretrained(weights_dir)
        self.config = self.model.config
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def get_patch_tokens(feat):
        """
        Slices off 1 CLS + 4 Register tokens from DINOv3's output.
        
        Register tokens are additional learned tokens (like the CLS token) added to the 
        input sequence. They act as a storage mechanism for global or redundant image 
        information, preventing the model from dumping this information into the spatial 
        patch tokens. This results in cleaner, artifact-free spatial feature maps.
        Because they do not correspond to spatial locations, we discard them for PatchCore.
        """
        return feat[:, 5:, :]

    @staticmethod
    def format_to_spatial_grid(feat):
        """
        Extracts patch tokens and reshapes them from a 1D sequence into a 2D spatial grid.
        
        For example, a typical DINOv3 output feature map might be [B, 201, C].
        After slicing off the 5 special tokens (CLS + 4 Register tokens), we have 196 patch tokens.
        These 196 tokens represent actual spatial regions and are reshaped into a 14x14 
        spatial grid [B, 14, 14, C]. Finally, the tensor is permuted to [B, C, H, W] 
        to match the expected format for PatchCore.
        """
        patch_tokens = DinoV3Backbone.get_patch_tokens(feat)
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        return patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
