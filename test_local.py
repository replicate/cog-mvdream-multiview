import json
import base64
import requests


resp = requests.post(
    "http://localhost:5000/predictions",
    data=json.dumps(
        {
            "input": {
                "prompt":"an green astronaut riding a horse",
                "image_size": 256,
                "num_frames":8,
                "num_inference_steps":45,
            }
        }
    ),
)

output = resp.json()["output"]
image_list = output["images"]

for i, (image_base64, camera_matrix, azimuth_angle) in enumerate(zip(output["images"], output["camera_matrices"], output["azimuth_angles"])):
    print(camera_matrix)
    print(azimuth_angle)
    with open(f"response_image{i}.png", "wb") as fh:
        fh.write(base64.b64decode(image_base64.replace("data:image/png;base64,", "")))