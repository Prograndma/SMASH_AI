import argparse
import torch
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_vit_regressor import CustomViTRegressor
import numpy as np
import torch.optim as optim
import os
import time
from my_secrets import WORKING_DIR

BATCH_SIZE = 32
THRESHOLD = 50

NUM_CONTROLLER_OUTPUTS = 18             # There are 18 outputs in a normal controller
NUM_CONTROLLER_SUBTRACTING = 0          # We might remove some buttons to simplify. (dpad, l&r pressure, maybe y)

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

VIT_BASE_DIR = f"{WORKING_DIR}/vit/{WHICH_GAME}/{DATASET_TYPE}_no_shuffle_differentiable_loss"
VIT_BASE_FILENAME = f"{VIT_BASE_DIR}/checkpoints"
SAVE_PATH_DATASET = f"{WORKING_DIR}/dataset/{WHICH_GAME}/{DATASET_TYPE}"

if WHICH_GAME == "smash":
    dataset_name = "smash"
elif WHICH_GAME == "kart":
    dataset_name = "mario_kart"
else:
    raise Exception("Choose a valid game")

if DATASET_TYPE == "full":
    HUGGING_FACE_DATASET_KEY = f"tomc43841/public_{dataset_name}_full_dataset"
elif DATASET_TYPE == "balanced":
    HUGGING_FACE_DATASET_KEY = f"tomc43841/public_{dataset_name}_balanced_dataset"
elif DATASET_TYPE == "medium":
    HUGGING_FACE_DATASET_KEY = f"tomc43841/public_{dataset_name}_medium_dataset"
else:
    HUGGING_FACE_DATASET_KEY = f"tomc43841/public_{dataset_name}_little_dataset"


EPOCH_FILE = f"{VIT_BASE_DIR}/epochs.txt"
LOSSES_FILE = f"{VIT_BASE_DIR}/train_losses.txt"
VALIDATION_LOSSES_FILE = f"{VIT_BASE_DIR}/val_losses.txt"
TEST_LOSS_FILE = f"{VIT_BASE_DIR}/test_losses.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=4,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--run_test",
        type=int,
        default=0,
        help="Default=0, if set to 1 don't train, don't validate, just load the desired model and test it"
    )
    parser.add_argument(
        "--load_epoch",
        type=int,
        default=None,
        help="Default=None, if set the program will attempt to load that specific model."
    )

    args = parser.parse_args()
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    return args


def train_test_valid_split(dataset, valid_percent, test_percent, seed=42):
    # This is important! There should not be any amount of randomizing when splitting the dataset! That is because
    # there is likely a high amount of data correlation between frames that are close to each other. For example,
    # imagine I am holding a button for 10 frames straight, the images are probably very similar and the desired model
    # output is identical. If one of those frames appears in the training, another in validation and another still in
    # the testing set that is testing the model on things that are incredibly similar to it's training data.


    # Split once into train and (valid and test)

    train_test_valid = dataset.train_test_split(test_size=valid_percent + test_percent, seed=seed, shuffle=False)

    # Split (valid and test) into train and test - call the train of this part validation
    new_test_size = test_percent / (valid_percent + test_percent)
    test_valid = train_test_valid['test'].train_test_split(test_size=new_test_size, seed=seed, shuffle=False)

    # gather all the pieces to have a single DatasetDict
    # It is now appropriate to shuffle the datasets.
    train_test_valid_dataset = DatasetDict({
        'train': train_test_valid['train'].shuffle(seed=seed),
        'validation': test_valid['train'].shuffle(seed=seed),
        'test': test_valid['test'].shuffle(seed=seed),
        })
    return train_test_valid_dataset


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_loss = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, inp, target):
        cross = self.cross_loss(inp[:12], target[:12])
        mse = self.mse(inp[12:], target[12:])
        loss = cross + mse
        return loss


class CombinedDifferentiableLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(CombinedDifferentiableLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.device = device
        self.binary_mask = torch.tensor([1] * 12 + [0] * 6, dtype=torch.float32, device=self.device)
        self.continuous_mask = torch.tensor([0] * 12 + [1] * 6, dtype=torch.float32, device=self.device)

    def forward(self, predictions, targets):
        # Calculate the binary and continuous losses. We pass in the masked values to do the proper type of loss.
        binary_loss = self.bce_loss(predictions * self.binary_mask, targets * self.binary_mask)
        continuous_loss = self.mse_loss(predictions * self.continuous_mask, targets * self.continuous_mask)

        # binary_loss = self.bce_loss(predictions, targets)
        # continuous_loss = self.mse_loss(predictions, targets)


        # Combine the losses
        total_loss = binary_loss + continuous_loss
        return total_loss


def _train(model,
           num_epochs_to_train,
           loss_values,
           validation_loss_values,
           train_loader,
           val_loader,
           optimizer,
           scheduler,
           objective,
           device='cuda'):
    total_epochs = 0
    total_batches = 0

    vl = []
    print(f"Batches in Epoch: {len(train_loader)}")
    if os.path.isfile(EPOCH_FILE):
        with open(f"{EPOCH_FILE}", "r") as f:
            for line in f:
                total_epochs = int(line.strip())
                break
            if total_epochs > 0:
                model.update_model_from_checkpoint(f"{total_epochs - 1}")
                print(f"LOADED MODEL FROM CHECKPOINT {total_epochs - 1}")
    for epoch in range(num_epochs_to_train):
        # GETTING EPOCH NUMBER
        if not os.path.isfile(EPOCH_FILE):
            with open(f"{EPOCH_FILE}", "w") as f:
                f.write("0\n")
        else:
            with open(f"{EPOCH_FILE}", "r") as f:
                for line in f:
                    total_epochs = int(line.strip())
                    break

        # DO AN EPOCH
        batch_losses = []
        print(f"DOING EPOCH {total_epochs}")
        model.train()
        for batch, (x, y_truth) in enumerate(train_loader):  # learn
            if total_batches == 1:  # or total_batches == 2 or total_batches == 10:
                start = time.time()
                print(f"Just did {total_batches} batches", end=" | | ")
                vl = []
                with torch.no_grad():
                    for batch_index, (x_l, y_l) in enumerate(val_loader):
                        x_v, y_v = x_l['pixel_values'].to(device), y_l.to(device)
                        vl.append(objective(model(x_v), y_v).item())
                    val = np.mean(vl)
                    validation_loss_values.append((total_batches, val))
                    vl = []
                print(f"Just finished testing against validation set. It took {time.time() - start}")
            if total_batches % (len(train_loader) // 10) == 0:
                print(f"Just did {total_batches} batches")  # , end=" | | ")
                # vl = []
                # with torch.no_grad():
                #     for batch_index, (x_l, y_l) in enumerate(val_loader):
                #         x_v, y_v = x_l['pixel_values'].to(device), y_l.to(device)
                #         vl.append(objective(model(x_v), y_v).item())
                #     val = np.mean(vl)
                #     validation_loss_values.append((total_batches, val))
                #     vl = []
                # print("Just finished testing against validation set")
            x, y_truth = x['pixel_values'].to(device), y_truth.to(device)

            optimizer.zero_grad()
            y_hat = model(x)

            loss = objective(y_hat, y_truth)
            loss.backward()
            batch_losses.append(loss.item())
            vl.append(loss.item())
            optimizer.step()
            del loss

            total_batches += 1
        scheduler.step()
        # CHECKPOINT MODEL
        model.save(total_epochs)

        # EPOCH OVER, INCREMENT EPOCHS AND SAVE NEW VALUE
        total_epochs += 1
        with open(f"{EPOCH_FILE}", "w") as f:
            f.write(str(total_epochs) + "\n")

        # UPDATE TRAINING SET LOSSES
        loss_values.append((total_epochs, np.mean(batch_losses)))

        with open(f"{LOSSES_FILE}", "a+") as f:
            loss = loss_values[-1]
            f.write(str(loss) + "\n")
            # for loss in loss_values:
            #     f.write(str(loss) + "\n")

        with open(f"{VALIDATION_LOSSES_FILE}", "w") as f:
            val_loss = validation_loss_values[-1]
            f.write(str(val_loss) + "\n")
            # for val_loss in validation_loss_values:
            #     f.write(str(val_loss) + "\n")

    with torch.no_grad():
        for batch_index, (x_v, y_v) in enumerate(val_loader):
            x_v, y_v = x_v['pixel_values'].to(device), y_v.to(device)
            vl.append(objective(model(x_v), y_v).item())
        val = np.mean(vl)
        validation_loss_values.append((total_batches, val))
    with open(f"{VALIDATION_LOSSES_FILE}", "w") as f:
        for val_loss in validation_loss_values:
            f.write(str(val_loss) + "\n")


def is_interesting_target(target):
    if target[0] == 1 or \
            target[1] == 1 or \
            target[2] == 1 or \
            target[3] == 1 or \
            target[4] == 1 or \
            target[5] == 1 or \
            target[6] == 1 or \
            target[7] == 1 or \
            target[8] == 1 or \
            target[9] == 1 or \
            target[10] == 1 or \
            target[11] == 1:
        return 1
    if target[12] >= THRESHOLD or \
            target[13] >= THRESHOLD or \
            abs(target[14] - 127.5) >= THRESHOLD or \
            abs(target[15] - 127.5) >= THRESHOLD or \
            abs(target[16] - 127.5) >= THRESHOLD or \
            abs(target[17] - 127.5) >= THRESHOLD:
        return 1
    return 0


def _infer(model, test_loader, objective, device='cuda'):
    test_losses = []
    for x_v, y_v in test_loader:
        x_v, y_v = x_v['pixel_values'].to(device), y_v.to(device)
        test_losses.append(objective(model(x_v), y_v).item())
        val = np.mean(test_losses)
        test_losses.append(val)

    with open(f"{TEST_LOSS_FILE}", "w") as f:
        for val_loss in test_losses:
            f.write(str(val_loss) + "\n")


def main(args):
    if os.path.isdir(f"{SAVE_PATH_DATASET}/train"):
        dataset = load_from_disk(SAVE_PATH_DATASET)
        try:
            model = CustomViTRegressor(VIT_BASE_FILENAME, cull=True)
        except OSError:
            model = CustomViTRegressor(VIT_BASE_FILENAME, should_load_from_disk=False, cull=True)

    else:
        dataset = load_dataset(HUGGING_FACE_DATASET_KEY)
        dataset = train_test_valid_split(dataset['train'], .15, .15)
        dataset.save_to_disk(SAVE_PATH_DATASET)
        _ = CustomViTRegressor(VIT_BASE_FILENAME, should_load_from_disk=False)
        return
    device = "cuda"

    custom_data_collator = CustomViTRegressor.custom_data_collator_function()

    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)
    val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)
    test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)

    model = model.to(device)

    losses = []
    val_losses = []

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=args.max_train_steps)

    # objective = torch.nn.MSELoss()
    # objective = torch.nn.CrossEntropyLoss()
    objective = CombinedDifferentiableLoss(device=device)

    _train(model, args.max_train_steps, losses, val_losses, train_loader,
           val_loader, optimizer, scheduler, objective, device=device)


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
