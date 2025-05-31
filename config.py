import os
import torch

class CFG:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_DEVICES = torch.cuda.device_count()
    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 30
    EPOCHS = 30
    BATCH_SIZE = (
        256 if torch.cuda.device_count() < 2 
        else (256 * torch.cuda.device_count())
    )
    TEST_SIZE = 0.15
    LR = 0.001
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.1
    APPLY_SHUFFLE = True
    SEED = 768
    HEIGHT = 224
    WIDTH = 224
    CHANNELS = 3
    IMAGE_SIZE = (224, 224, 3)