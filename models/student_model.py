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
        
        #off-the-shelf CLIP ViT-B/16
        #'WithProjection' to get the final 512-D visual embeddings directly
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        self.model.to(self.device)
        
        #freeze the model, (no fine-tuning for baseline)
        self.model.eval()
        
        #not tracking gradients
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
        #ove images to the correct hardware spot
        images = images.to(self.device)

        with torch.no_grad(): #save memory
            #pass through ViT
            outputs = self.model(images)
            raw_embeddings = outputs.image_embeds

        #L2 Normalization 
        #(ensures all vectors have a length of 1)
        normalized_embeddings = F.normalize(raw_embeddings, p=2, dim=1)

        return normalized_embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension D."""
        return 512

