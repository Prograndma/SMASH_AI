from datasets import load_from_disk
import torch
from transformers import ViTImageProcessor
from torch.utils.data import DataLoader
from vit import CustomViTRegressor, custom_data_collator_function
from vgg import CustomSimpleImageRegressor
from vgg import custom_data_collator_function as vgg_collator
from custom_image_processor import CustomImageProcessor

WORKING_DIR = "/home/thomas/PycharmProjects/SMASH_AI"
WHAT_WE_WORKING_ON = "balanced"
SAVE_PATH_DATASET = f"{WORKING_DIR}/dataset/{WHAT_WE_WORKING_ON}"

BATCH_SIZE = 1


def binary_accuracy(y_true, y_hat):
    # Extract the first 12 values from both tensors
    y_true_binary = y_true[:, :12]
    y_hat_binary = y_hat[:12]

    # Calculate accuracy
    correct_predictions = torch.sum(y_true_binary * y_hat_binary)
    total_positive_instances = torch.sum(y_true_binary)

    if total_positive_instances == 0:
        accuracy = 0.0    # Avoid division by zero
    else:
        accuracy = (correct_predictions / total_positive_instances).item()

    return accuracy


def binary_precision(y_true, y_hat):
    # Extract the first 12 values from both tensors
    y_true_binary = y_true[:, :12]
    y_hat_binary = y_hat[:12]

    # Calculate precision
    true_positives = torch.sum(y_true_binary * y_hat_binary)
    total_positive_predictions = torch.sum(y_hat_binary)

    if total_positive_predictions == 0:
        precision = 0.0  # Avoid division by zero
    else:
        precision = (true_positives / total_positive_predictions).item()

    return precision


def binary_recall(y_true, y_hat):
    # Extract the first 12 values from both tensors
    y_true_binary = y_true[:, :12]
    y_hat_binary = y_hat[:12]

    # Calculate recall
    true_positives = torch.sum(y_true_binary * y_hat_binary)
    total_actual_positives = torch.sum(y_true_binary)

    if total_actual_positives == 0:
        recall = 0.0  # Avoid division by zero
    else:
        recall = (true_positives / total_actual_positives).item()

    return recall


def acc_check(model, dataloader, threshold):
    # accuracy of first 12: %accurate, or just % of trues in y_true that are also true.
    # accuracy of last 6: % within 50 of the truth value?
    binary_accuracies = []
    bin_true_acc = []
    in_range = []
    recalls = []
    precisions = []
    for epoch, (x, y_true) in enumerate(dataloader):

        x, y_true = x['pixel_values'].to(device), y_true.to(device)
        y_hat = model(x)
        # y_hat = torch.cat((torch.round(y_hat[0, :12]), y_hat[0, 12:]))
        y_hat = torch.cat(((y_hat[0, :12] > threshold).float(), y_hat[0, 12:]))

        binary_accuracies.append((y_hat[:12] == y_true[:, :12]).float().mean().item())
        distance_last_six = torch.abs(y_true[:, 12:] - y_hat[12:])
        in_range_count = torch.sum(distance_last_six < .196)     # 50/255
        in_range.append(in_range_count.item()/6)

        # If first 12 are not all 0, get accuracy of 1s
        if not torch.all(y_hat[:12] == 0):
            precisions.append(binary_precision(y_true, y_hat))

        if not torch.all(y_true[:, :12] == 0):
            bin_true_acc.append(binary_accuracy(y_true, y_hat))

            recalls.append(binary_recall(y_true, y_hat))
            # print("y_true: ", y_true)
            # print("y-hat: ", y_hat)
            # # print("bin_acc ", binary_accuracies)
            # print("bin_true_acc ", bin_true_acc[-1])
            # print("in_rance ", in_range)

        if epoch > 100000:
            break

    return bin_true_acc, binary_accuracies, in_range, recalls, precisions


def main(model, data_loader):
    # threshes = [.1, .11, .12, .13, .14, .15]
    threshes = [.14]
    for threshold in threshes:
        bin_acc_true, bin_acc, in_range, recalls, precisions = acc_check(model, data_loader, threshold)
        print(f"FOR A THRESHOLD OF {threshold}")
        # print(sum(bin_acc) / len(bin_acc))
        # print(sum(bin_acc_true) / len(bin_acc_true))
        print(sum(in_range) / len(in_range))
        # print(f"RECALL   : {sum(recalls) / len(recalls)}")
        # print(f"PRECISION: {sum(precisions) / len(precisions)}\n")


if __name__ == "__main__":
    dataset = load_from_disk(SAVE_PATH_DATASET)
    custom_data_collator = custom_data_collator_function(ViTImageProcessor())

    val_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)

    device = 'cuda'
    print("FOR SUPER COMPUTER TRAINED MODEL")
    model = CustomViTRegressor()
    model.update_model_from_checkpoint("13")
    model.to(device)
    main(model, val_loader)
    print("################################################\n\n")

    #
    print("FOR OUR MODEL TRAINED ON OUR NEW LOSS FUNCTION:")
    model = CustomViTRegressor()
    model.update_model_from_checkpoint("newloss_2")
    model.to(device)
    main(model, val_loader)
    print("################################################\n\n")

    print("FOR OUR VGG MODEL:")

    dataset = load_from_disk(SAVE_PATH_DATASET)
    custom_data_collator = vgg_collator(CustomImageProcessor())

    val_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, collate_fn=custom_data_collator)

    model = CustomSimpleImageRegressor()
    model.update_model_from_checkpoint("newloss_2")
    model.to(device)
    main(model, val_loader)
