import argparse
import torch
from datasets import load_from_disk
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_vgg_regressor import CustomVGGRegressor
import numpy as np
import torch.optim as optim
import os
import time
from my_secrets import WORKING_DIR
from dataset_constructor import DatasetConstructor

BATCH_SIZE = 1
THRESHOLD = 50

DATASET_TYPE = "balanced"
# DATASET_TYPE = "full"
# DATASET_TYPE = "medium"
# DATASET_TYPE = "little"
WHICH_GAME = "smash"
# WHICH_GAME = "kart"

if WHICH_GAME == "smash":
    dataset_name = "smash"
elif WHICH_GAME == "kart":
    dataset_name = "mario_kart"
else:
    raise Exception("Choose a valid game")

VGG_BASE_DIR = f"{WORKING_DIR}/vgg/{WHICH_GAME}/{DATASET_TYPE}_culled"
VGG_BASE_FILENAME = f"{VGG_BASE_DIR}/checkpoints"
SAVE_PATH_DATASET = f"{WORKING_DIR}/dataset/{WHICH_GAME}/{DATASET_TYPE}_extended"

if WHICH_GAME == "smash":
    dataset_name = "smash"
elif WHICH_GAME == "kart":
    dataset_name = "mario_kart"
else:
    raise Exception("Choose a valid game")


EPOCH_FILE = f"{VGG_BASE_DIR}/epochs.txt"
LOSSES_FILE = f"{VGG_BASE_DIR}/train_losses.txt"
VALIDATION_LOSSES_FILE = f"{VGG_BASE_DIR}/val_losses.txt"
TEST_LOSS_FILE = f"{VGG_BASE_DIR}/test_losses.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    args = parser.parse_args()
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    return args



class CombinedDifferentiableLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(CombinedDifferentiableLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.device = device
        self.binary_mask = torch.tensor([1] * 5 + [0] * 6, dtype=torch.float32, device=self.device)
        self.continuous_mask = torch.tensor([0] * 5 + [1] * 6, dtype=torch.float32, device=self.device)

    def forward(self, predictions, targets):
        # Calculate the binary and continuous losses. We pass in the masked values to do the proper type of loss.
        binary_loss = self.bce_loss(predictions * self.binary_mask, targets * self.binary_mask)
        continuous_loss = self.mse_loss(predictions * self.continuous_mask, targets * self.continuous_mask)

        total_loss = binary_loss + continuous_loss
        return total_loss


def _train(model, num_epochs_to_train, train_loader, val_loader, optimizer, scheduler, objective, device='cuda'):
    print(f"Batches in Epoch: {len(train_loader)}")

    if not os.path.isfile(EPOCH_FILE):
        with open(f"{EPOCH_FILE}", "w") as f:
            f.write("0\n")
    for epoch in range(num_epochs_to_train):
        # DO AN EPOCH
        print(f"DOING EPOCH {epoch}")
        model.train()
        write_loss = -1.0
        print("|##########|")
        print("|", end="")
        for batch, (x, y_truth) in enumerate(train_loader):  # learn
            if batch % (len(train_loader) // 10) == 0:
                print("#", end="")
            x, y_truth = x['pixel_values'].to(device), y_truth.to(device)

            optimizer.zero_grad()
            y_hat = model(x)

            loss = objective(y_hat, y_truth)
            loss.backward()
            optimizer.step()
            write_loss = loss.item()
            del loss
        print("|")

        print(f"Just did {epoch} epochs")
        with torch.no_grad():
            temp_val = []
            for batch_index, (x_l, y_l) in enumerate(val_loader):
                x_v, y_v = x_l['pixel_values'].to(device), y_l.to(device)
                temp_val.append(objective(model(x_v), y_v).item())
            val = np.mean(temp_val)

        scheduler.step()
        # CHECKPOINT MODEL
        model.save(epoch)

        # EPOCH OVER, SAVE NEW VALUE
        with open(f"{EPOCH_FILE}", "w") as f:
            f.write(str(epoch) + "\n")

        with open(f"{LOSSES_FILE}", "a+") as f:
            f.write(f"{epoch}, {write_loss}\n")

        with open(f"{VALIDATION_LOSSES_FILE}", "a+") as f:
            f.write(f"{epoch}, {val}\n")



def is_interesting_target(target):
    if target[0] == 1 or \
        target[1] == 1 or \
        target[2] == 1 or \
        target[3] == 1 or \
        target[5] == 1:
        return 1
    if target[12] >= THRESHOLD or \
        target[13] >= THRESHOLD or \
        abs(target[14] - 127.5) >= THRESHOLD or \
        abs(target[15] - 127.5) >= THRESHOLD or \
        abs(target[16] - 127.5) >= THRESHOLD or \
        abs(target[17] - 127.5) >= THRESHOLD:
        return 1
    return 0


def main(args):
    if os.path.isdir(f"{SAVE_PATH_DATASET}/train"):
        dataset = load_from_disk(SAVE_PATH_DATASET)
    else:
        constructor = DatasetConstructor(which_game=WHICH_GAME, dataset_type=DATASET_TYPE, use_suffixes=True)
        constructor.download_dataset()
        return
    device = "cuda"

    dataset = dataset.remove_columns(["Y", "DPadUp", "DPadDown", "DPadLeft", "DPadRight", "L", "R"])
    custom_data_collator = CustomVGGRegressor.custom_data_collator_function()

    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)
    val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)
    model = CustomVGGRegressor(VGG_BASE_FILENAME)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=args.max_train_steps)

    objective = CombinedDifferentiableLoss(device=device)
    # objective = torch.nn.MSELoss()
    # objective = torch.nn.CrossEntropyLoss()

    _train(model, args.max_train_steps, train_loader, val_loader, optimizer, scheduler, objective, device=device)


if __name__ == "__main__":
    now = time.time()
    try:
        main(parse_args())
        print("##################################################")
        print(f"Time taken = {time.time() - now}")
    except Exception as e:
        print("##################################################")
        print(f"Time taken = {time.time() - now}")
        raise e
