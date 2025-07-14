import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from typing import Union
from torch_snippets import *

class CableCabinetClassifier(nn.Module):
    idx_to_class = {  # ✅ define once, used by all instances
        0: 'BC', 1: 'BC K.M.skab', 2: 'CP1', 3: 'CP3', 4: 'CP4', 5: 'CP6', 6: 'KSE09',
        7: 'KSE12', 8: 'KSE15', 9: 'KSE18', 10: 'KSE21', 11: 'KSE27', 12: 'KSE36/45',
        13: 'Kabeldon CDC440/460/420', 14: 'Kabeldon KSIP423', 15: 'Kabeldon KSIP433',
        16: 'Kabeldon KSIP443', 17: 'Kabeldon KSIP463/KSIP483', 18: 'Kombimodul 2M',
        19: 'Kombimodul 3M', 20: 'Kombimodul 4M', 21: 'MEL1', 22: 'MEL2', 23: 'MEL3',
        24: 'MEL4', 25: 'NU', 26: 'PK20', 27: 'PK35', 28: 'PK48', 29: 'SC'
    }

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(0).to(self.device)
        outputs = self.model(x)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        top_probs, top_indices = torch.topk(probs, k=3)
        top_classes = [self.idx_to_class[i.item()] for i in top_indices]
        return list(zip(top_classes, top_probs.cpu().numpy().tolist()))

    def predict(self, input: Union[str, Image.Image]):
        if isinstance(input, str):
            img = Image.open(input).convert("RGB")
        elif isinstance(input, Image.Image):
            img = input.convert("RGB")
        else:
            raise ValueError("Input must be a path or PIL image")

        img = self.transform(img)
        result = self.forward(img)
        return {"top_3_predictions": result}

    predict_from_path = predict
    predict_from_image = predict