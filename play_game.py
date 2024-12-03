import argparse

from transformers.models.clap.convert_clap_original_pytorch_to_hf import processor

from custom_vit_regressor import CustomViTRegressor
from binary_regressor import CustomBinaryImageRegressor
from vgg import CustomSimpleImageRegressor
from custom_image_processor import CustomImageProcessor
from my_secrets import WORKING_DIR

class PlayGame:
    def __init__(self, discriminator, model, discriminator_image_processor, model_image_processor,
                 outfile, image_source, device, use_same_processor_for_discriminator_and_model=True):
        self.discriminator = discriminator
        self.model = model
        self.outfile = outfile
        self.device = device
        self.image_source = image_source
        self.model_image_processor = model_image_processor
        self.discriminator_image_processor = discriminator_image_processor
        self.use_same_processor_for_discriminator_and_model = use_same_processor_for_discriminator_and_model

    def play(self, num_frames=-1):
        frames_processed = 0
        while frames_processed != num_frames:
            frames_processed += 1

            input_image = self.image_source.get()
            discriminator_input_image = self.discriminator_image_processor.preprocess(input_image, return_tensors="pt").to(self.device)
            should_press_buttons = self.discriminator.play(discriminator_input_image)
            if should_press_buttons:
                if self.use_same_processor_for_discriminator_and_model:
                    model_input_image = discriminator_input_image
                else:
                    model_input_image = self.model_image_processor.preprocess(input_image, return_tensors="pt").to(self.device)
                buttons_to_press = self.get_buttons_pressed(model_input_image)

    def get_buttons_pressed(self, processed_input_image):
        # TODO: finish this
        return self.model.play(processed_input_image)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--num_frames",
                        type=int,
                        default=-1,
                        help="The amount of frames to play until the game stops. ")
    parser.add_argument("--model_checkpoint",
                        type=int,
                        required=True,
                        help="The name of the model checkpoint to load.",)
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="The name of the model_dir, it should have a checkpoints dir inside of it.")
    parser.add_argument("--game",
                        type=str,
                        choices=["smash", "kart"],
                        required=True,)
    parser.add_argument("--model_type",
                        type=str,
                        choices=["vit", "vgg"],
                        default="vit",
                        help="Which model type to load")
    parser.add_argument("--model_threshold",
                        type=float,
                        default=.15,
                        help="The threshold to set the model to, you should know the optimal threshold.")
    parser.add_argument("--discriminator_threshold",
                        type=float,
                        default=.5,
                        help="The threshold to set the discriminator to, you should know the optimal threshold.")
    parser.add_argument("--discriminator_type",
                        type=str,
                        choices=["vit_binary", "vgg_binary"],
                        default="vit_binary",
                        help="Which model type to load")
    parser.add_argument("--discriminator_checkpoint",
                        type=int,
                        required=True,
                        help="The name of the model checkpoint to load.", )
    parser.add_argument("--discriminator_dir",
                        type=str,
                        required=True,
                        help="The name of the model_dir, it should have a checkpoints dir inside of it.")

    args = parser.parse_args()

    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    return args



def main(args):
    if args['model_type'] == "vit":
        model = CustomViTRegressor(f"{WORKING_DIR}/{args['model_type']}/{args['game']}/{args['model_dir']}/checkpoints")
        model.update_model_from_checkpoint(args['checkpoint'])
        model_processor = CustomImageProcessor()
    if args['model_type'] == "vgg":
        raise Exception("Not yet implemented")

    if args['discriminator_type'] == "vit_binary":
        # discriminator = CustomViTBinaryRegressor(f"{WORKING_DIR}/{args['model_type']}/{args['game']}/{args['model_dir']}/checkpoints")
        # discriminator.update_model_from_checkpoint(args['checkpoint'])
        raise Exception("Not yet implemented")
    if args['discriminator_type'] == "vgg":
        discriminator = CustomBinaryImageRegressor(f"{WORKING_DIR}/{args['model_type']}/{args['game']}/{args['model_dir']}/checkpoints")
        discriminator.update_model_from_checkpoint(args['checkpoint'])
        discriminator_processor = CustomImageProcessor()


    # game_player = PlayGame(discriminator, model, )
    # game_player = PlayGame(discriminator, model, discriminator_processor, model_processor, )

if __name__ == "__main__":
    command_args = parse_args()
    main(command_args)
