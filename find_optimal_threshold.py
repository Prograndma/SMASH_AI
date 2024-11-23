import argparse

from datasets import load_from_disk
from torch.utils.data import DataLoader
from custom_vit_regressor import CustomViTRegressor
from custom_vgg_regressor import CustomVGGRegressor
import time
from my_secrets import WORKING_DIR

WHAT_WE_WORKING_ON = "smash/balanced_extended"
SAVE_PATH_DATASET = f"{WORKING_DIR}/dataset/{WHAT_WE_WORKING_ON}"

NUM_CONTINUOUS = 6  # Amount of continuous buttons
NUM_BUTTS = 12      # Amount of buttons
NUM_CULLED_BUTTS = 5
BUTT_NAMES = ["Start", "A", "B", "X", "Y", "Z", "DPadUp", "DPadDown", "DPadLeft", "DPadRight", "L", "R",
              "LPressure", "RPressure", "XAxis", "YAxis", "CXAxis", "CYAxis",]
CULLED_BUTT_NAMES = ["Start", "A", "B", "X", "Z", "LPressure", "RPressure", "XAxis", "YAxis", "CXAxis", "CYAxis",]


def get_butt_names(culled):
    if not culled:
        return BUTT_NAMES
    return CULLED_BUTT_NAMES



class FindOptimalThreshold:
    def __init__(self, name_index, culled):
        self.targets = []
        self.model_results = []
        self.name = get_butt_names(culled)[name_index]
        self.reordered_targets = None
        self.sorted_guesses = None
        self.from_the_left = None
        self.from_the_right = None
        self.total_one_guesses = 0

    def add(self, target, model_result):
        self.targets.append(target)
        self.model_results.append(model_result)
        self.reordered_targets = None
        self.sorted_guesses = None
        self.from_the_left = None
        self.from_the_right = None

    def _reorganize(self):
        zip_sorted = sorted(zip(self.model_results, self.targets), key=lambda pair: pair[0])
        self.sorted_guesses = [x for x, _ in zip_sorted]
        self.reordered_targets = [x for _, x in zip_sorted]
        self.total_one_targets = sum(self.reordered_targets)
        self.from_the_left = []
        num_ones_seen = 0
        self.from_the_right = []
        num_zeroes_seen = 0
        for i in range(len(self.reordered_targets)):
            if self.reordered_targets[i] == 1:
                num_ones_seen += 1
            if self.reordered_targets[-1 - i] == 0:
                num_zeroes_seen += 1
            self.from_the_left.append(num_ones_seen)
            self.from_the_right.append(num_zeroes_seen)
        self.from_the_right.reverse()

    def find_optimal_f1(self):
        if self.reordered_targets is None or self.sorted_guesses is None or self.from_the_left is None or self.from_the_right is None:
            self._reorganize()
        if self.total_one_targets == 0:
            return 100, 1, 1, 1,
        previous_f1 = 0.0
        have_found_f1 = False
        best_threshold = -111
        best_precision = 0.0
        best_recall = 0.0
        for i in range(len(self.reordered_targets)):
            total_positive = len(self.sorted_guesses[i:])
            true_positive = sum(self.reordered_targets[i:])

            if true_positive == 0:  # Not good enough.
                continue
            false_negative = sum(self.reordered_targets[:i])

            precision = true_positive / total_positive
            recall = true_positive / (true_positive + false_negative)

            if (precision + recall) == 0:
                continue
            f1 = (2 * precision * recall) / (precision + recall)
            if f1 >= previous_f1:
                previous_f1 = f1
                best_threshold = self.sorted_guesses[i]
                best_recall = recall
                best_precision = precision
                have_found_f1 = True

        if not have_found_f1:
            raise Exception("Didn't find a good f1!")
        return best_threshold, best_precision, best_recall, previous_f1


    def find_threshold_and_performance(self):
        if self.reordered_targets is None or self.sorted_guesses is None or self.from_the_left is None or self.from_the_right is None:
            self._reorganize()
        if self.total_one_targets == 0:
            return 100, 1, 1, 1, 1, 1
        found_good_thresh = False
        for i in range(len(self.reordered_targets)):
            if self.from_the_right[i] < self.from_the_left[i]:
                found_good_thresh = True
                optimal_threshold = self.sorted_guesses[i]
                total_positive = len(self.sorted_guesses[i:])
                true_positive = sum(self.reordered_targets[i:])
                if true_positive == 0:          # Not good enough to even press the button.
                    print(f"{self.name} Is bad. ")
                    return 100, 0.0, 0.0, 0.0, 0.0, 0.0
                true_negative = len(self.reordered_targets[:i]) - sum(self.reordered_targets[:i])
                false_positive = total_positive - true_positive
                false_negative = sum(self.reordered_targets[:i])
                if total_positive == 0:
                    raise Exception(f"For {self.name}: total_positive = 0")
                if (true_positive + false_negative) == 0:
                    raise Exception(f"For {self.name}: true_positive + false_negative = 0")

                precision = true_positive / total_positive
                recall = true_positive / (true_positive + false_negative)

                if (precision + recall) == 0:
                    raise Exception(f"For {self.name}: precision + recall = 0")
                if (true_negative + false_positive) == 0:
                    raise Exception(f"For {self.name}: true_negative + false_positive = 0")
                f1 = (2 * precision * recall) / (precision + recall)
                accuracy = (true_positive + true_negative) / len(self.reordered_targets)
                specificity = true_negative / (true_negative + false_positive)
                break
        ###
        if not found_good_thresh:
            raise Exception("Didn't find a good threshold!")
        return optimal_threshold, precision, recall, f1, accuracy, specificity



def acc_check(model, dataloader, device, num_butts):
    culled = False
    if num_butts < NUM_BUTTS:
        culled = True
    threshold_finders: list[FindOptimalThreshold] =  [FindOptimalThreshold(i, culled) for i in range(num_butts)]
    continuous_variable_errors = []
    print("|###########|")
    print("|", end="")
    for batch, (x, y_true) in enumerate(dataloader):
        if batch % (len(dataloader) // 10) == 0:
            print("#", end="")

        x, y_true = x['pixel_values'].to(device), y_true.to(device)[0]
        y_hat = model(x)[0]

        diffs = y_hat[num_butts:NUM_CONTINUOUS + num_butts] - y_true[num_butts:NUM_CONTINUOUS + num_butts]
        continuous_variable_errors.append(abs(diffs).tolist())

        for i in range(num_butts):
            threshold_finders[i].add(int(y_true[i].item()), y_hat[i].item())
    print("|")

    means = [0] * NUM_CONTINUOUS
    for instance in continuous_variable_errors:
        for i in range(len(instance)):
            means[i] += instance[i]

    for i in range(len(means)):
        means[i] /= len(continuous_variable_errors)
    thresholds = []

    for threshold_finder in threshold_finders:
        thresholds.append(threshold_finder.find_threshold_and_performance())
        thresholds[-1] += threshold_finder.find_optimal_f1()
    results = []
    optimized_precision = []
    optimized_recall = []
    print("RESULTS")
    for i, butt_name in enumerate(get_butt_names(culled)):
        if i < num_butts:
            current_threshold = thresholds[i][0]
            precision = thresholds[i][1]
            recall = thresholds[i][2]
            f1 = thresholds[i][3]
            accuracy = thresholds[i][4]
            specificity = thresholds[i][5]
            optimized_f1_threshold = thresholds[i][6]
            optimized_f1_precision = thresholds[i][7]
            optimized_f1_recall = thresholds[i][8]
            optimized_f1 = thresholds[i][9]
            print(f"{butt_name.upper()}\n"
                  f"threshold: {current_threshold}\n"
                  f"precision = {precision}\n"
                  f"recall = {recall}\n"
                  f"f1 = {f1}\n"
                  f"accuracy = {accuracy}\n"
                  f"specificity = {specificity}\n"
                  f"optimized f1 threshold: {optimized_f1_threshold}\n"
                  f"optimized f1 precision: {optimized_f1_precision}\n"
                  f"optimized f1 recall: {optimized_f1_recall}\n"
                  f"optimized f1: {optimized_f1}\n")
            results.append(precision)
            optimized_recall.append(optimized_f1_recall)
            optimized_precision.append(optimized_f1_precision)
        else:
            print(f"{butt_name.upper()}: average error = {means[i - num_butts] * 255}")
            results.append(means[i - num_butts] * 255)
    return results, optimized_precision, optimized_recall


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--culled",
        type=bool,
        default=False,
        help="Are we removing the un-needed buttons?",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["vgg", "vit", "vgg_binary", "vit_binary"],
        default="vit",
        help="Type of model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="The folder, after the model_type, that hold checkpoints"
    )
    parser.add_argument(
        "--first_checkpoint",
        type=int,
        default=0,
        help="Which checkpoint to start loading"
    )
    parser.add_argument(
        "--last_checkpoint",
        type=int,
        required=True,
        help="the last checkpoint you want to load."
    )

    args = parser.parse_args()
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    # TODO: FIX CULLED
    # args.culled = False
    dataset = load_from_disk(SAVE_PATH_DATASET)
    my_butts = NUM_BUTTS
    if args.culled:
        dataset = dataset.remove_columns(["Y", "DPadUp", "DPadDown", "DPadLeft", "DPadRight", "L", "R"])
        my_butts = NUM_CULLED_BUTTS

    if args.model_type == "vit":
        custom_data_collator = CustomViTRegressor.custom_data_collator_function(args.culled)
        model_constructor = CustomViTRegressor
    elif args.model_type == "vgg":
        custom_data_collator = CustomVGGRegressor.custom_data_collator_function()
        model_constructor = CustomVGGRegressor
    else:
        raise NotImplementedError("Not yet implemented")

    val_loader = DataLoader(dataset["validation"], collate_fn=custom_data_collator)

    device = 'cuda'
    binary_precisions = "Epoch, "
    continuous_errors = "Epoch, "
    for butt_name in get_butt_names(args.culled)[:my_butts -1]:
        binary_precisions += f"{butt_name}, "
    binary_precisions += f"{get_butt_names(args.culled)[my_butts - 1]}\n"
    optimized_binary_precisions = binary_precisions
    optimized_binary_recall = binary_precisions

    for butt_name in get_butt_names(args.culled)[my_butts: NUM_CONTINUOUS + my_butts -1]:
        continuous_errors += f"{butt_name}, "
    continuous_errors += f"{get_butt_names(args.culled)[-1]}\n"


    for i in range(args.first_checkpoint, args.last_checkpoint + 1):
        start = time.time()
        base_dir = f"{WORKING_DIR}/{args.model_type}/{args.model_path}"
        print(f"FOR {i} EPOCHS")
        model = model_constructor(base_filename=f"{base_dir}/checkpoints", cull=args.culled)
        model.update_model_from_checkpoint(f"{i}")
        model.to(device)
        result, opt_precision, opt_recall = acc_check(model, val_loader, device, my_butts)
        binary_precisions += f"{i}, "
        continuous_errors += f"{i}, "
        optimized_binary_precisions += f"{i}, "
        optimized_binary_recall += f"{i}, "

        for thing in result[:my_butts -1]:
            binary_precisions += f"{thing}, "
        binary_precisions += f"{result[my_butts - 1]}\n"

        for thing in opt_precision[:my_butts -1]:
            optimized_binary_precisions += f"{thing}, "
        optimized_binary_precisions += f"{opt_precision[my_butts - 1]}\n"

        for thing in opt_recall[:my_butts -1]:
            optimized_binary_recall += f"{thing}, "
        optimized_binary_recall += f"{opt_recall[my_butts - 1]}\n"

        for thing in result[my_butts: NUM_CONTINUOUS + my_butts - 1]:
            continuous_errors += f"{thing}, "
        continuous_errors += f"{result[-1]}\n"

        print(f"Time Elapse: {time.time() - start}")
        print("################################################\n\n")

    print("When recall and precision are the same:")
    print(binary_precisions)
    print("The Average Error for the continuous variables")
    print(continuous_errors)
    print("Precision scores when optimizing for F1 score")
    print(optimized_binary_precisions)
    print("recalls scores when optimizing for F1 score")
    print(optimized_binary_recall)


if __name__ == "__main__":
    main()