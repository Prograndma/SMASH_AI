import torch
import torch.nn as nn
from transformers import ViTModel
import os


WHAT_WE_WORKING_ON = "balanced"
VIT_BASE_DIR = f"vit/{WHAT_WE_WORKING_ON}"
VIT_BASE_FILENAME = f"{VIT_BASE_DIR}/checkpoints"

HUGGING_FACE_PRETRAINED_VIT = "google/vit-base-patch16-224-in21k"


class CustomViTRegressor(nn.Module):
    def __init__(self, should_load_from_disk=True, base_filename=VIT_BASE_FILENAME, base_dir=VIT_BASE_DIR):
        super(CustomViTRegressor, self).__init__()
        if should_load_from_disk:
            self.base_model = ViTModel.from_pretrained(f"{base_filename}/pretrained")
        else:
            self.base_model = ViTModel.from_pretrained(HUGGING_FACE_PRETRAINED_VIT)
            self.base_model.save_pretrained(f"{base_filename}/pretrained")
        self.regressor = nn.Linear(self.base_model.config.hidden_size, 18)  # amount of controller inputs
        self.base_filename = base_filename
        self.base_dir = base_dir

    def forward(self, x):
        features = self.base_model(x)
        predictions = self.regressor(features.pooler_output)
        return predictions

    def saved_model_exists(self, checkpoint):
        if not os.path.exists(f"{self.base_filename}/{checkpoint}"):
            return False
        return True

    def update_model_from_checkpoint(self, checkpoint):
        if not self.saved_model_exists(checkpoint):
            return "no saved model exists"

        path = f"{self.base_filename}/{checkpoint}"

        loaded = torch.load(path)
        return self.load_state_dict(loaded)

    def save(self, epoch):
        torch.save(self.state_dict(), f"{self.base_filename}/{epoch}")
