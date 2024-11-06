import os
import numpy as np
from vit import CombinedDifferentiableLoss


class BestModelCheckpoint:
    def __init__(self, model_class, model_checkpoints_folder, validation_loader, objective, validation_losses_outfile,
                 device="cuda"):
        self.model_class = model_class
        if os.path.isdir(model_checkpoints_folder):
            self.model_checkpoints_folder = model_checkpoints_folder
        else:
            self.model_checkpoints_folder = None
            raise Exception(f"{model_checkpoints_folder} Is not a folder, can not initialize BestModelCheckpoint")
        self.validation_loader = validation_loader
        self.objective = objective
        self.validation_losses_outfile = validation_losses_outfile
        self.device = device

    def run_tests(self):
        checkpoints = (file for file in os.listdir(self.model_checkpoints_folder)
                       if os.path.isfile(os.path.join(self.model_checkpoints_folder, file)))
        checkpoints = sorted(checkpoints, key=int)
        validation_losses = []
        print("Starting validation testing on all checkpoints")
        print(f"batches per checkpoint: {len(self.validation_loader)}")
        with torch.no_grad():
            for checkpoint in checkpoints:
                print(f"Checkpoint {checkpoint} starting")
                model = self.model_class(base_dir=None, base_filename=self.model_checkpoints_folder)
                model.update_model_from_checkpoint(checkpoint)
                model = model.to(self.device)

                temp_val = []
                for batch_index, (x_l, y_l) in enumerate(self.validation_loader):
                    x_v, y_v = x_l['pixel_values'].to(self.device), y_l.to(self.device)
                    temp_val.append(self.objective(model(x_v), y_v).item())
                val = np.mean(temp_val)
                validation_losses.append((checkpoint, val))


        # with open(f"{self.validation_losses_outfile}", "w") as f:
        #     for val_loss in validation_losses:
        #         f.write(str(val_loss) + "\n")
        for val_loss in validation_losses:
            print(str(val_loss) + "\n")


if __name__ == "__main__":
    from datasets import load_from_disk
    from custom_vit_regressor import CustomViTRegressor
    import torch
    from torch.utils.data import DataLoader

    dataset_path = "/home/thomas/PycharmProjects/SMASH_AI/dataset/smash/balanced"
    dataset = load_from_disk(dataset_path)

    custom_data_collator = CustomViTRegressor.custom_data_collator_function()
    val_loader = DataLoader(dataset["validation"], batch_size=32, collate_fn=custom_data_collator)
    bmc = BestModelCheckpoint(model_class=CustomViTRegressor,
                              model_checkpoints_folder="/home/thomas/PycharmProjects/SMASH_AI/vit/smash/balanced_every_iter_16_batch/checkpoints",
                              validation_loader=val_loader,
                              objective=CombinedDifferentiableLoss(),
                              validation_losses_outfile="/home/thomas/PycharmProjects/SMASH_AI/vit/smash/balanced_every_iter_16_batch/val_losses.txt",
                              )
    bmc.run_tests()