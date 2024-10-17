import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
import os


HUGGING_FACE_PRETRAINED_VIT = "google/vit-base-patch16-224-in21k"


class CustomViTRegressor(nn.Module):
    """
        base_dir should take into account which game is being trained on and which dataset is being used
        base_filename should be where the checkpoints of the model are/should be
    """
    def __init__(self, base_dir, base_filename, should_load_from_disk=True):
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

    @staticmethod
    def custom_data_collator_function():
        def return_func(batch):
            images = [item["image"] for item in batch]

            inputs = ViTImageProcessor(images, return_tensors="pt")
            targets = torch.tensor([[
                item["Start"],
                item["A"],
                item["B"],
                item["X"],
                item["Y"],
                item["Z"],
                item["DPadUp"],
                item["DPadDown"],
                item["DPadLeft"],
                item["DPadRight"],
                item["L"],
                item["R"],
                item["LPressure"] / 255,
                item["RPressure"] / 255,
                item["XAxis"] / 255,
                item["YAxis"] / 255,
                item["CXAxis"] / 255,
                item["CYAxis"] / 255] for item in batch], dtype=torch.float32)

            return inputs, targets

        return return_func
