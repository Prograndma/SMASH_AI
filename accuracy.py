from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader

from custom_vit_regressor import CustomViTRegressor


WORKING_DIR = "/home/thomas/PycharmProjects/SMASH_AI"
WHAT_WE_WORKING_ON = "balanced"
SAVE_PATH_DATASET = f"{WORKING_DIR}/dataset/{WHAT_WE_WORKING_ON}"

BATCH_SIZE = 1


def find_num_perfect_matches(predictions, targets, threshold):

    for prediction, target in zip(predictions, targets):
        thresh_prediction = prediction[:12] > threshold


def find_best_threshold(predictions, targets):
    while(True):
        find_num_perfect_matches(predictions, targets, .5)

def main(model, dataloader):
    predictions = []
    targets = []
    for image, (frame, target) in enumerate(dataloader):
        with torch.no_grad():
            frame, target = frame['pixel_values'].to(device), target.to(device)
            prediction = model(frame)
            predictions.append(prediction.cpu().tolist())
            targets.append(target.cpu().tolist())
    num_buttons_perfect_match = 0
    # num_
    # for prediction, target in zip(predictions, targets):



if __name__ == "__main__":
    dataset = load_from_disk(SAVE_PATH_DATASET)
    custom_data_collator = CustomViTRegressor.custom_data_collator_function()

    val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)

    device = 'cuda'

    model = CustomViTRegressor(base_filename=None)
    # model.update_model_from_checkpoint("newloss_0")
    model.update_model_from_checkpoint("6")
    model.to(device)
    main(model, val_loader)
