import numpy as np
from PIL import ImageGrab
import math
from time import sleep
from PIL import Image
import os


game_cords = [122, 121, 925, 724]


if __name__ == "__main__":
    os.system("gnome-screenshot --file=this_directory.png")
    # screen_cap = np.array(ImageGrab.grab(bbox=game_cords))[:, :, 1]
    # screen_cap.show()
