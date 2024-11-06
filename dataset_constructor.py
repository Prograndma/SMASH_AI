from datasets import load_dataset, load_from_disk, concatenate_datasets
import os
from my_secrets import WORKING_DIR
from datasets import DatasetDict

EXISTING_DATASET_SUFFIXES = ["01", "02", "03", "04", "05", "06", "07", ]
ALLOWED_GAME_TYPES_AND_NAMES = {"smash": "smash",
                                "kart": "mario_kart"}
ALLOWED_DATASET_TYPES = ["full", "balanced", "medium", "little"]




class DatasetConstructor:
    def __init__(self, which_game="smash", dataset_type="balanced", use_suffixes=True):
        if which_game not in ALLOWED_GAME_TYPES_AND_NAMES.keys():
            raise Exception(f"Choose a valid game, to {which_game}")
        if dataset_type not in ALLOWED_DATASET_TYPES:
            raise Exception(f"Choose a valid dataset type, not {dataset_type}")
        self.which_game = which_game
        self.dataset_type = dataset_type
        self.use_suffixes = use_suffixes
        self.dataset_keys = self._get_dataset_key()
        self.save_path = self._get_save_path()

    def download_dataset(self):
        if os.path.isdir(f"{self.save_path}/train"):
            raise Exception(f"Dataset at {self.save_path} already exists!")

        loaded_datasets = []

        for i, ds_key in enumerate(self.dataset_keys):
            loaded_datasets.append(load_dataset(ds_key)['train'])
            print(f"##################\nLoaded {i + 1} of {len(self.dataset_keys)}\n##################")
        dataset = self._train_test_valid_split(loaded_datasets, .15, .15)
        dataset.save_to_disk(self.save_path)

    def _train_test_valid_split(self, datasets_to_combine, valid_percent, test_percent, seed=42):
        # This is important! There should not be any amount of randomizing when splitting the dataset! That is because
        # there is likely a high amount of data correlation between frames that are close to each other. For example,
        # imagine I am holding a button for 10 frames straight, the images are probably very similar and the desired model
        # output is identical. If one of those frames appears in the training, another in validation and another still in
        # the testing set that is testing the model on things that are incredibly similar to it's training data.


        # Split once into train and (valid and test)
        dataset = datasets_to_combine[0]
        if len(datasets_to_combine) > 1:
            dataset = concatenate_datasets(datasets_to_combine)

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

    def _get_dataset_key(self):

        dataset_name = ALLOWED_GAME_TYPES_AND_NAMES[self.which_game]

        if self.which_game == "kart":
            self.use_suffixes = False

        if self.dataset_type == "full":
            hugging_face_dataset_key = f"tomc43841/public_{dataset_name}_full_dataset"
            if self.use_suffixes:
                hugging_face_dataset_key += "_"
        elif self.dataset_type == "balanced":
            hugging_face_dataset_key = f"tomc43841/public_{dataset_name}_balanced_dataset"
            if self.use_suffixes:
                hugging_face_dataset_key += "_"
        elif self.dataset_type == "medium":
            self.use_suffixes = False
            hugging_face_dataset_key = f"tomc43841/public_{dataset_name}_medium_dataset"
        else:
            self.use_suffixes = False
            hugging_face_dataset_key = f"tomc43841/public_{dataset_name}_little_dataset"
        keys = []

        if self.use_suffixes:
            for suffix in EXISTING_DATASET_SUFFIXES:
                keys.append(hugging_face_dataset_key + suffix)
        else:
            keys.append(hugging_face_dataset_key)

        return keys

    def _get_save_path(self):
        if self.use_suffixes:
            return f"{WORKING_DIR}/dataset/{self.which_game}/{self.dataset_type}_extended"
        return f"{WORKING_DIR}/dataset/{self.which_game}/{self.dataset_type}"
