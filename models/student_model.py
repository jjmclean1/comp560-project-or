import torch
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection

class StudentModel:
    def __init__(self, device: str = "cuda"):
        """
        Initialize your model.

        Args:
            device: Target device ("cuda" or "cpu")
        """
        self.device = device
        
        #Baseline off-the-shelf CLIP ViT-B/16
        #'WithProjection' to get the final 512-D visual embeddings directly
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        self.model.to(self.device)
        
        #Freeze the model, (no fine-tuning for baseline)
        self.model.eval()
        
        #safety: explicitly tell PyTorch not to track gradients
        #save memory on Tinker.
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: (B, 3, H, W) tensor, normalized with ImageNet stats
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

        Returns:
            (B, D) tensor of L2-normalized embeddings
        """
        #Move images to the correct hardware
        images = images.to(self.device)

        with torch.no_grad(): #Disable gradient calculations to save memory during inference
            # Pass images through the ViT
            outputs = self.model(images)
            raw_embeddings = outputs.image_embeds

        # Apply L2 Normalization 
        # (This ensures all vectors have a length of 1, making Cosine Similarity work perfectly)
        normalized_embeddings = F.normalize(raw_embeddings, p=2, dim=1)

        return normalized_embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension D."""
        # The standard OpenAI ViT-B/16 outputs 512-dimensional vectors
        return 512

