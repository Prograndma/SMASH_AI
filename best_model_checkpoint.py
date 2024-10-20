import os
import numpy as np


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
        checkpoints = sorted(checkpoints)
        for checkpoint in checkpoints:
            # TODO: There needs to be two different validation_losses lists or something.
            model = self.model_class(base_dir=None, base_filename=self.model_checkpoints_folder, should_load_from_disk=False)
            model.update_model_from_checkpoint(checkpoint)
            validation_losses = []

            for batch_index, (x_l, y_l) in enumerate(self.validation_loader):
                x_v, y_v = x_l['pixel_values'].to(self.device), y_l.to(self.device)
                validation_losses.append(self.objective(model(x_v), y_v).item())
            val = np.mean(validation_losses)
            validation_loss_values.append((checkpoint, val))
            validation_losses = []


            with open(f"{self.validation_losses_outfile}", "w") as f:
                for val_loss in test_losses:
                    f.write(str(val_loss) + "\n")