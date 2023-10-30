import os
import random
from cog import BasePredictor, Input, Path, BaseModel
import torch
from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from typing import List
import file_utils


os.environ["HF_HOME"] = os.environ["HUGGINGFACE_HUB_CACHE"] = "/src/hf-cache"

MVDREAM_CHECKPOINT_PATH = "./mvdream/weights/sd-v2.1-base-4view.pt"
MVDREAM_CONFIG_PATH = "./mvdream/configs/sd-v2-base.yaml"
MODEL_FILES_MAP = {
    "MVDREAM-SD2.1": {
        "url": "https://weights.replicate.delivery/default/mvdream/sd-v2.1-base-4view.tar",
        "cache_dir": "./mvdream/weights",
    },
    "CLIP-ViT-H-14-laion2B": {
        "url": "https://weights.replicate.delivery/default/mvdream/clip_vit_h14_laion2B.tar",
        "cache_dir": "./hf-cache/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K",
    },
}

# Download model weights if their cache directory doesn't exist
for k, v in MODEL_FILES_MAP.items():
    if not os.path.exists(v["cache_dir"]):
        file_utils.download_and_extract(url=v["url"], dest=v["cache_dir"])

class ModelOutput(BaseModel):
    images: List[Path] # List of generated images
    camera_matrices: List[List[List[float]]] # List of camera matrices
    azimuth_angles: List[float] # List of azimuth angles

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instiantiate the model
        config = OmegaConf.load(MVDREAM_CONFIG_PATH)
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(
            torch.load(MVDREAM_CHECKPOINT_PATH, map_location=self.device)
        )
        self.model.device = self.device
        self.model.eval()
        self.model.to(self.device)

        # Instantiate the sampler
        self.sampler = DDIMSampler(self.model)

    def set_seed(self, seed):
        random.seed(seed)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def predict(
        self,
        prompt: str = Input(
            description="A prompt to condition the model",
            default="an astronaut riding a horse",
        ),
        image_size: int = Input(
            description="The size of the generated image",
            choices=[128, 256, 512, 1024],
            default=256,
        ),
        num_frames: int = Input(
            description="The number of frames to generate", ge=1, le=32, default=4
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        camera_elevation: int = Input(
            description="The elevation of the camera", ge=0, le=90, default=15
        ),
        camera_azimuth: int = Input(
            description="The azimuth of the camera", ge=0, le=360, default=90
        ),
        camera_azimuth_span: int = Input(
            description="The span of the azimuth of the camera",
            ge=0,
            le=360,
            default=360,
        ),
        seed: int = Input(
            description="The seed to use for the generation. If not specified, a random seed will be used",
            default=None,
        ),
    ) -> ModelOutput:
        self.set_seed(seed)

        # pre-compute camera matrices
        batch_size = num_frames
        camera, camera_matrices, azimuth_list = get_camera(
            num_frames,
            elevation=camera_elevation,
            azimuth_start=camera_azimuth,
            azimuth_span=camera_azimuth_span,
        )
        camera = camera.to(self.device)

        # Add the suffix to the prompt
        prompt = prompt + ", 3d asset"

        # Conditioning on the input prompt and camera matrices
        c = self.model.get_learned_conditioning([prompt]).to(self.device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc = self.model.get_learned_conditioning([""]).to(self.device)
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        # Sampling
        shape = [4, image_size // 8, image_size // 8]
        samples, _ = self.sampler.sample(
            S=num_inference_steps,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc_,
            eta=0.0,
            x_T=None,
        )
        image_tensors = self.model.decode_first_stage(samples)

        # Tensor to image list
        image_tensors = torch.clamp((image_tensors + 1.0) / 2.0, min=0.0, max=1.0)
        image_arrays = 255.0 * image_tensors.permute(0, 2, 3, 1).cpu().numpy()
        image_list = list(image_arrays.astype(np.uint8))

        # Real Shit
        output_images = []
        for i, image in enumerate(image_list):
            fname = f"/tmp/result_{i}.png"
            Image.fromarray(image).save(fname)
            output_images.append(Path(fname))
        
        output = ModelOutput(
            images = output_images,
            camera_matrices = camera_matrices,
            azimuth_angles = azimuth_list
        )
        
        return output