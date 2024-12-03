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
        self.thresholds = {}
        self.butt_names = {}
        self.butt_names_list = []
        self.all_butts = {}
        self.num_butts: int = 0
        self.num_continuous = 6
        self.cull = cull
        self.base_filename = base_filename
        self.set_butt_names()

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

    def forward(self, x):
        features = self.base_model(x)
        predictions = self.regressor(features.pooler_output)
        return predictions

    def set_threshold(self, thresholds: [float], checkpoint=None):
        for i, threshold in enumerate(thresholds):
            if i > self.num_butts:
                break
            self.thresholds[self.butt_names[i]] = threshold
        if checkpoint is None:
            return
        with open(f"{self.base_filename}\\{checkpoint}.txt", "w") as f:
            for thresh in self.thresholds:
                f.write(f"{self.thresholds[thresh]}, ")

    def play(self, inputs):
        if len(self.thresholds.keys()) == 0:
            raise Exception("Model can not play without set thresholds! Please set the the thresholds with the "
                            ".set_threshold function")
        with torch.no_grad():
            features = self.base_model(**inputs)
            predictions = self.regressor(features.pooler_output)[0]
        butt_presses = {}

        for i in range(self.num_butts):
            butt = self.butt_names[i]
            butt_presses[butt] = predictions[i].item() > self.thresholds[butt]

        for i in range(self.num_butts, self.num_butts + self.num_continuous):
            butt = self.butt_names[i]
            output = predictions[i].item()
            butt_presses[butt] = output
            # if butt in ["XAxis", "YAxis", "CXAxis", "CYAxis"]:
            butt_presses[butt] *= 255

        for un_butt in self.all_butts:
            curr_butt = self.all_butts[un_butt]
            if curr_butt not in self.butt_names_list:
                butt_presses[curr_butt] = False
        return butt_presses

    def saved_model_exists(self, checkpoint):
        if not os.path.exists(f"{self.base_filename}/{checkpoint}"):
            return False
        return True

    def update_model_from_checkpoint(self, checkpoint):
        if not self.saved_model_exists(checkpoint):
            return "no saved model exists"

        path = f"{self.base_filename}/{checkpoint}"

        loaded = torch.load(path)
        if os.path.exists(f"{self.base_filename}\\{checkpoint}.txt"):
            set_thresh = []
            with open(f"{self.base_filename}\\{checkpoint}.txt", "r") as f:
                for line in f.readlines():
                    threshes = line.split(',')
                    for thresh in threshes:

                        thresh = thresh.strip()
                        if len(thresh) == 0:
                            continue
                        thresh = float(thresh)
                        set_thresh.append(thresh)

            self.set_threshold(set_thresh, None)

        return self.load_state_dict(loaded)

    def save(self, epoch):
        torch.save(self.state_dict(), f"{self.base_filename}/{epoch}")

    def set_butt_names(self):
        self.num_butts = 12
        self.butt_names_list = ["Start",
                                "A",
                                "B",
                                "X",
                                "Y",
                                "Z",
                                "DPadUp",
                                "DPadDown",
                                "DPadLeft",
                                "DPadRight",
                                "L",
                                "R",
                                "LPressure",
                                "RPressure",
                                "XAxis",
                                "YAxis",
                                "CXAxis",
                                "CYAxis"]
        self.butt_names = {0: "Start",
                           1: "A",
                           2: "B",
                           3: "X",
                           4: "Y",
                           5: "Z",
                           6: "DPadUp",
                           7: "DPadDown",
                           8: "DPadLeft",
                           9: "DPadRight",
                           10: "L",
                           11: "R",
                           12: "LPressure",
                           13: "RPressure",
                           14: "XAxis",
                           15: "YAxis",
                           16: "CXAxis",
                           17: "CYAxis"}
        self.all_butts = self.butt_names
        if self.cull:
            self.num_butts = 5
            self.butt_names_list = ["Start",
                                    "A",
                                    "B",
                                    "X",
                                    "Z",
                                    "LPressure",
                                    "RPressure",
                                    "XAxis",
                                    "YAxis",
                                    "CXAxis",
                                    "CYAxis"]
            self.butt_names = {0: "Start",
                               1: "A",
                               2: "B",
                               3: "X",
                               4: "Z",
                               5: "LPressure",
                               6: "RPressure",
                               7: "XAxis",
                               8: "YAxis",
                               9: "CXAxis",
                               10: "CYAxis"}

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
