import argparse
import torch
from datasets import load_dataset, load_from_disk
from custom_image_processor import CustomImageProcessor
from datasets import DatasetDict
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import os
import time
from my_secrets import WORKING_DIR
from custom_vgg_regressor import CustomVGGRegressor


BATCH_SIZE = 1
THRESHOLD = 50

NUM_CONTROLLER_OUTPUTS = 18             # There are 18 outputs in a normal controller
NUM_CONTROLLER_SUBTRACTING = 0          # We might remove some buttons to simplify. (dpad, l&r pressure, maybe y)

WHAT_WE_WORKING_ON = "full"
GAME = "smash"
VGG_BASE_DIR = f"{WORKING_DIR}/vgg/{GAME}/{WHAT_WE_WORKING_ON}"
VGG_BASE_FILENAME = f"{VGG_BASE_DIR}/checkpoints"
SAVE_PATH_DATASET = f"{WORKING_DIR}/dataset/{WHAT_WE_WORKING_ON}"

if WHAT_WE_WORKING_ON == "full":
    HUGGING_FACE_DATASET_KEY = "tomc43841/public_smash_full_dataset"
elif WHAT_WE_WORKING_ON == "balanced":
    HUGGING_FACE_DATASET_KEY = "tomc43841/public_smash_balanced_dataset"
elif WHAT_WE_WORKING_ON == "medium":
    HUGGING_FACE_DATASET_KEY = "tomc43841/public_smash_medium_dataset"
else:
    HUGGING_FACE_DATASET_KEY = "tomc43841/public_smash_little_dataset"

EPOCH_FILE = f"{VGG_BASE_DIR}/epochs.txt"
LOSSES_FILE = f"{VGG_BASE_DIR}/train_losses.txt"
VALIDATION_LOSSES_FILE = f"{VGG_BASE_DIR}/val_losses.txt"
TEST_LOSS_FILE = f"{VGG_BASE_DIR}/test_losses.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3,
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
    # Split once into train and (valid and test)

    train_test_valid = dataset.train_test_split(test_size=valid_percent + test_percent, seed=seed)

    # Split (valid and test) into train and test - call the train of this part validation
    new_test_size = test_percent / (valid_percent + test_percent)
    test_valid = train_test_valid['test'].train_test_split(test_size=new_test_size, seed=seed)

    # gather all the pieces to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
        'train': train_test_valid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test'],
        })
    return train_test_valid_dataset


def _train(model,
           num_epochs_to_train,
           loss_values,
           validation_loss_values,
           train_loader,
           val_loader,
           optimizer,
           objective,
           device='cuda'):
    total_epochs = 0
    total_batches = 0

    vl = []
    for epoch in range(num_epochs_to_train):
        # GETTING EPOCH NUMBER
        if not os.path.isdir(EPOCH_FILE):
            with open(f"{EPOCH_FILE}", "w") as f:
                f.write("0\n")
        else:
            with open(f"{EPOCH_FILE}", "r") as f:
                for line in f:
                    total_epochs = int(line.strip())
                    break

        # DO AN EPOCH
        batch_losses = []
        print("DOING AN EPOCH")
        model.train()
        for batch, (x, y_truth) in enumerate(train_loader):  # learn
            if total_batches == 5 or total_batches == 50 or total_batches == 500:
                print(f"Just did {total_batches} batches")
                vl = []
                with torch.no_grad():
                    for batch_index, (x_l, y_l) in enumerate(val_loader):
                        x_v, y_v = x_l['pixel_values'].to(device), y_l.to(device)
                        vl.append(objective(model(x_v), y_v).item())
                    val = np.mean(vl)
                    validation_loss_values.append((total_batches, val))
                    vl = []
            if batch % 5000 == 0:
                print(f"Just did {total_batches} batches")
                vl = []
                with torch.no_grad():
                    for batch_index, (x_l, y_l) in enumerate(val_loader):
                        x_v, y_v = x_l['pixel_values'].to(device), y_l.to(device)
                        vl.append(objective(model(x_v), y_v).item())
                    val = np.mean(vl)
                    validation_loss_values.append((total_batches, val))
                    vl = []
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

        # CHECKPOINT MODEL
        model.save(total_epochs)

        # EPOCH OVER, INCREMENT EPOCHS AND SAVE NEW VALUE
        total_epochs += 1
        with open(f"{EPOCH_FILE}", "w") as f:
            f.write(str(total_epochs) + "\n")

        # UPDATE TRAINING SET LOSSES
        loss_values.append((total_epochs, np.mean(batch_losses)))

        with open(f"{LOSSES_FILE}", "a+") as f:
            for loss in loss_values:
                f.write(str(loss) + "\n")

        with open(f"{VALIDATION_LOSSES_FILE}", "w") as f:
            for val_loss in validation_loss_values:
                f.write(str(val_loss) + "\n")

    with torch.no_grad():
        for batch_index, (x_v, y_v) in enumerate(val_loader):
            x_v, y_v = x_v['pixel_values'].to(device), y_v.to(device)
            vl.append(objective(model(x_v), y_v).item())
        val = np.mean(vl)
        validation_loss_values.append((total_batches, val))
    with open(f"{VALIDATION_LOSSES_FILE}", "w") as f:
        for val_loss in validation_loss_values:
            f.write(str(val_loss) + "\n")


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


def custom_data_binary_collator_function(processor):
    def return_func(batch):
        images = [item["image"] for item in batch]

        inputs = processor(images, return_tensors="pt")
        targets = torch.tensor([[
            item["Start"],
            item["A"],
            item["B"],
            item["X"],
            # item["Y"],
            item["Z"],
            # item["DPadUp"],
            # item["DPadDown"],
            # item["DPadLeft"],
            # item["DPadRight"],
            # item["L"],
            # item["R"],
            item["LPressure"] / 255,
            item["RPressure"] / 255,
            item["XAxis"] / 255,
            item["YAxis"] / 255,
            item["CXAxis"] / 255,
            item["CYAxis"] / 255] for item in batch], dtype=torch.float32)

        targets = torch.tensor([[is_interesting_target(item)] for item in targets], dtype=torch.float32)

        return inputs, targets

    return return_func


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
    else:
        dataset = load_dataset(HUGGING_FACE_DATASET_KEY)
        dataset = train_test_valid_split(dataset['train'], .15, .15)
        dataset.save_to_disk(SAVE_PATH_DATASET)
    device = "cuda"

    custom_data_collator = custom_data_collator_function(CustomImageProcessor())

    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)
    val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)
    test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)

    # custom_binary_collator = custom_data_binary_collator_function(CustomImageProcessor())
    #
    # train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, collate_fn=custom_binary_collator)
    # val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, collate_fn=custom_binary_collator)

    model = CustomVGGRegressor(VGG_BASE_FILENAME).to(device)
    # model = CustomBinaryImageRegressor().to(device)

    losses = []
    val_losses = []

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    objective = torch.nn.MSELoss()

    _train(model, args.max_train_steps, losses, val_losses, train_loader,
           val_loader, optimizer, objective, device=device)

    # training_args = TrainingArguments(VGG_BASE_FILENAME, overwrite_output_dir=True,
    #                                   per_device_train_batch_size=BATCH_SIZE, max_steps=10)
    # trainer = Trainer(model, training_args, custom_data_collator, dataset["train"], dataset['validation'])
    # trainer.train()
    # model.update_model_from_checkpoint()


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
