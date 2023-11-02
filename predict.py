# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import os
import math
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers import (DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler
)

MODEL_CACHE = "cache"

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        unet = UNet2DConditionModel.from_pretrained(MODEL_CACHE, subfolder="unet", in_channels=9, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
        pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL_CACHE, unet=unet, safety_checker = None
        )
        self.pipe = pipe.to("cuda")

    def scale_down_image(self, image_path, max_size):
        #Open the Image
        image = Image.open(image_path)
        #Get the Original width and height
        width, height = image.size
        # Calculate the scaling factor to fit the image within the max_size
        scaling_factor = min(max_size/width, max_size/height)
        # Calaculate the new width and height
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = image.resize((new_width, new_height))
        cropped_image = self.crop_center(resized_image)
        return cropped_image, width, height

    def crop_center(self, pil_img):
        img_width, img_height = pil_img.size
        crop_width = self.base(img_width)
        crop_height = self.base(img_height)
        return pil_img.crop(
                (
                    (img_width - crop_width) // 2,
                    (img_height - crop_height) // 2,
                    (img_width + crop_width) // 2,
                    (img_height + crop_height) // 2)
                )

    def base(self, x):
        return int(8 * math.floor(int(x)/8))

    def predict(
        self,
        image: Path = Input(description="Input image"),
        mask: Path = Input(description="Mask image"),
        prompt: str = Input(
            description="Input prompt",
            default="a tabby cat, high resolution, sitting on a park bench",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        ),
        strength: float = Input(description="strength/weight", ge=0, le=1, default=1.0),
        steps: int = Input(description=" num_inference_steps", ge=0, le=100, default=30),
        guidance_scale: float = Input(
            description="Guidance scale", ge=0, le=10, default=7.5
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER_ANCESTRAL",
        ),
        use_karras_sigmas: bool = Input(description="use karras sigmas or not", default=False),
        seed: int = Input(description="Leave blank to randomize",  default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed == 0) or (seed == None):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config, use_karras_sigmas=use_karras_sigmas)
        
        print("Scheduler:", scheduler)
        print("Using karras sigmas:", use_karras_sigmas)
        print("Using seed:", seed)

        r_image, image_width, image_height  = self.scale_down_image(image, 1280)
        r_mask, mask_width, mask_height = self.scale_down_image(mask, 1280)
        width, height = r_image.size
        output_image = self.pipe(
            prompt=prompt,
            image=r_image,
            mask_image=r_mask,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            generator=generator,
        ).images[0]
        
        output_image = output_image.resize((image_width, image_height), Image.LANCZOS)
        out_path = Path(f"/tmp/output.png")
        output_image.save(out_path)
        return  out_path
