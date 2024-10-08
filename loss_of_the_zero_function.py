import numpy as np
import os
from datasets import load_from_disk
import torch
from custom_image_processor import CustomImageProcessor
from torch.utils.data import DataLoader


def get_dirs(which_database):
    working_dir = "/home/thomas/PycharmProjects/SMASH_AI"

    num_controller_outputs = 18  # There are 18 outputs in a normal controller
    num_controller_subtracting = 0  # We might remove some buttons to simplify. (d-pad, l&r pressure, maybe y)

    save_path_dataset = f"{working_dir}/dataset/{which_database}"
    return save_path_dataset, num_controller_outputs - num_controller_subtracting


def zero_function(objective, target, num_nothing):
    zeroes = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .5, .5, .5, .5]], dtype=torch.float32)
    return objective(zeroes, target)


def custom_data_collator_function(processor):
    def return_func(batch):
        images = [item["image"] for item in batch]

        inputs = processor(images, return_tensors="pt")
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


def main(which_database):
    objective = torch.nn.MSELoss()
    database_path, num_nothing = get_dirs(which_database)
    if os.path.isdir(f"{database_path}/validation"):
        dataset = load_from_disk(database_path)
        custom_data_collator = custom_data_collator_function(CustomImageProcessor())
        val_loader = DataLoader(dataset["validation"], batch_size=1, collate_fn=custom_data_collator)
    else:
        raise Exception("NON VALID DATABASE DIR")
    validation_loss_values = []
    vl = []
    for batch_index, (_, y_v) in enumerate(val_loader):
        vl.append(zero_function(objective, y_v, num_nothing))
        if batch_index % 500 == 0:
            val = np.mean(vl)
            validation_loss_values.append(val)
            vl = []
    print(np.mean(validation_loss_values))


if __name__ == "__main__":
    databases = ["full", "balanced", "medium", "small"]
    main(databases[1])
