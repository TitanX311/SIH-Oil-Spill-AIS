import requests
import json

url = "http://localhost:8000/predict"

with open("/media/titanx/disk/01_Train_Val_Oil_Spill_images/01339.tif", "rb") as f:
# with open("/home/titanx/Pictures/Gemini_Generated_Image_r8zwzgr8zwzgr8zw.png", "rb") as f:
    response = requests.post(
        url,
        files={
            "file": (
                "/media/titanx/disk/01_Train_Val_Oil_Spill_images/01339.tif",
                f,
                "image/tiff"
            )
        }
    )

print(json.dumps(response.json(), indent=4))