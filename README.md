# Cog MVDream Multi-View 
This is an implementation of MVDream's text-to-multi-view image generation module as a [Cog](https://github.com/replicate/cog) model. See the [paper](https://arxiv.org/abs/2308.16512), [original repository](https://github.com/bytedance/MVDream) and this [Replicate model](https://replicate.com/adirik/mvdream-multi-view).

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of MVDream to [Replicate](https://replicate.com).


## API Usage
You will need to have Cog and Docker installed on your local to run predictions. To use MVDream, simply describe the scene in natural language, and set stable diffusion generation parameters, camera elevation and/or azimuth angle span if you wish. The model will generate a consistent set of images from different views. API has the following inputs:

- prompt: What you want to generate expressed in natural language
- image_size: Width and height of the generated images. allowed values are 128, 256, 512, 1024. Note, larger is better, but slower.
- num_frames: Number of views to generate.
- num_inference_steps: Number of diffusion steps. Higher values will lead to better quality, but slower generation.
- guidance_scale: How much to guide the generation process with the prompt. Higher values will lead to generation that is closer to the prompt, but less diverse or maybe of lower quality.
- camera_elevation: Elevation angle of the camera.
- camera_azimuth: Azimuth angle of the camera in the first view.
- camera_azimuth_span: Total span of the azimuth angle. For example if the span is kept as 360 degrees and num_frames is set to 5 then in each view azimuth angle will be incremented by 360/5=72 degrees.
- seed: Random seed for the generation process. If not specified, a random seed will be used.


To run a prediction:
```bash
cog predict -i prompt="an astronaut riding a horse" -i image_size=512
```

To build the cog image and launch the API on your local:
```bash
cog run -p 5000 python -m cog.server.http
```

## References 
```
@article{shi2023MVDream,
  author = {Shi, Yichun and Wang, Peng and Ye, Jianglong and Mai, Long and Li, Kejie and Yang, Xiao},
  title = {MVDream: Multi-view Diffusion for 3D Generation},
  journal = {arXiv:2308.16512},
  year = {2023},
}
```