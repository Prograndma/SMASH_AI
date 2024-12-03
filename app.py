from custom_vit_regressor import CustomViTRegressor, ViTImageProcessor
from my_secrets import WORKING_DIR
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from base64 import b64decode as decode

class SyncBody(BaseModel):
    data: str
    model_config = {"arbitrary_types_allowed": True}


app = FastAPI()

CHECKPOINT = 9
CULL = True
BASE_DIR = f"{WORKING_DIR}\\vit\\smash\\balanced_culled\\checkpoints"
DEVICE = "cuda"


processor = ViTImageProcessor()
model = CustomViTRegressor(BASE_DIR, cull=CULL)
model.update_model_from_checkpoint(CHECKPOINT)
model.to(DEVICE)

@app.post("/sync")
def sync(width: int, height: int, body: SyncBody):
    data = body.data
    decoded: bytes = decode(data)
    image = Image.frombytes('RGBA', (width, height), decoded, 'raw').convert("RGB")
    inputs = processor(image, return_tensors="pt").to(DEVICE)

    return model.play(inputs)


if __name__ == "__main__":
    main_image = Image.open("C:\\Users\\User\\Downloads\\screen_cap.jpg").convert("RGB")
    main_inputs = processor(main_image, return_tensors="pt").to(DEVICE)

    main_result = model.play(main_inputs)
    print(main_result)
