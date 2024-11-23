import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
import os


HUGGING_FACE_PRETRAINED_VIT = "google/vit-base-patch16-224-in21k"


class CustomViTRegressor(nn.Module):
    """
        base_filename should be where the checkpoints of the model are/should be
    """
    def __init__(self, base_filename, cull, should_load_from_disk=True):
        self.threshold = None
        self.cull = cull
        super(CustomViTRegressor, self).__init__()
        if should_load_from_disk:
            self.base_model = ViTModel.from_pretrained(f"{base_filename}/pretrained")
        else:
            self.base_model = ViTModel.from_pretrained(HUGGING_FACE_PRETRAINED_VIT)
            self.base_model.save_pretrained(f"{base_filename}/pretrained")
        if self.cull:
            self.regressor = nn.Linear(self.base_model.config.hidden_size, 11)  # amount of culled controller inputs
        else:
            self.regressor = nn.Linear(self.base_model.config.hidden_size, 18)  # amount of controller inputs
        self.base_filename = base_filename

    def forward(self, x):
        features = self.base_model(x)
        predictions = self.regressor(features.pooler_output)
        return predictions

    def set_threshold(self, threshold: float):
        if threshold < 0.0 < 1.0:
            raise Exception("Model's threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    def play(self, image):
        if self.threshold is None:
            raise Exception("Model can not play without a set threshold! Please set the the threshold with the "
                            ".set_threshold function")
        with torch.no_grad():
            features = self.base_model(image)
            predictions = self.regressor(features.pooler_output)


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
    def custom_data_collator_function(cull):
        if cull:
            def return_func(batch):
                images = [item["image"] for item in batch]
                temp_processor = ViTImageProcessor()
                inputs = temp_processor(images, return_tensors="pt")
                targets = torch.tensor([[
                    item["Start"],
                    item["A"],
                    item["B"],
                    item["X"],
                    item["Z"],
                    item["LPressure"] / 255,
                    item["RPressure"] / 255,
                    item["XAxis"] / 255,
                    item["YAxis"] / 255,
                    item["CXAxis"] / 255,
                    item["CYAxis"] / 255] for item in batch], dtype=torch.float32)

                return inputs, targets
        else:
            def return_func(batch):
                images = [item["image"] for item in batch]
                temp_processor = ViTImageProcessor()
                inputs = temp_processor(images, return_tensors="pt")
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
