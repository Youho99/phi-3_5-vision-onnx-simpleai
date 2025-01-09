from dataclasses import dataclass
from typing import Optional, Union

from simple_ai.api.grpc.chat.server import LanguageModel  # type: ignore
import onnxruntime_genai as og  # type: ignore
import base64
from io import BytesIO
from PIL import Image  # type: ignore
import os
import glob
# import torch
# import time


SNAPSHOT_MODEL_PATH = "~/.cache/huggingface/hub/models--microsoft--Phi-3.5-vision-instruct-onnx/snapshots/"


def get_model_onnx_folder():
    try:
        cache_dir = os.path.expanduser(SNAPSHOT_MODEL_PATH)
        snapshots = sorted(glob.glob(os.path.join(cache_dir, "*")), key=os.path.getmtime, reverse=True)
        
        if not snapshots:
            raise FileNotFoundError("No snapshots found.")
        
        latest_snapshot = snapshots[0]

        specific_folder = os.path.join(latest_snapshot, "gpu/gpu-int4-rtn-block-32")
        
        # if torch.cuda.is_available():
        #     specific_folder = os.path.join(latest_snapshot, "gpu/gpu-int4-rtn-block-32")
        # else:
        #     specific_folder = os.path.join(latest_snapshot, "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4")
        
        return specific_folder
    
    except Exception as e:
        raise RuntimeError(f"Error while retrieving ONNX folder: {str(e)}")
        

@dataclass(unsafe_hash=True)
class Phi35VisionModel(LanguageModel):

    og_model: og.Model = None
    og_processor: any = None
    og_tokenizer_stream: any = None

    def __init__(self):
        self.og_model = og.Model(get_model_onnx_folder())
        self.og_processor = self.og_model.create_multimodal_processor()
        self.og_tokenizer_stream = self.og_processor.create_stream()

    def chat(
        self,
        chatlog: list[list[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.9,
        top_p: int = 0.5,
        role: str = "assistant",
        n: int = 1,
        stream: bool = False,
        stop: Optional[Union[str, list]] = "",
        presence_penalty: float = 0.0,
        frequence_penalty: float = 0.0,
        logit_bias: Optional[dict] = {},
        *args,
        **kwargs,
    ):

        user_image, user_prompt = extract_message_data_openai(chatlog)  # only the last image

        image = None
        prompt = "<|user|>\n"

        if user_image is not None:
            if user_image.startswith("data:image"):
                user_image = save_base64_image(user_image, "tmp_img")

            image = og.Images.open(os.path.abspath(user_image))
            prompt += "<|image_1|>\n"

        prompt += f"{user_prompt}<|end|>\n<|assistant|>\n"
        inputs = self.og_processor(prompt, images=image)

        params = og.GeneratorParams(self.og_model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=3072)  # max_length en dur ??

        generator = og.Generator(self.og_model, params)

        answer = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            answer += self.og_tokenizer_stream.decode(new_token)

        # Delete the generator to free the captured graph before creating another one
        del generator

        if os.path.exists(user_image):
            os.remove(user_image)

        return [{"role": role, "content": answer}]


def extract_message_data_openai(
    messages,
):
    image_url = None
    text = None

    for message in messages:
        if isinstance(message["content"], str):
            text = message["content"]
        else:
            if message["role"] == "user":
                for content in message["content"]:
                    if isinstance(content, dict):
                        if content["type"] == "image_url":
                            image_url = content["image_url"]["url"]
                        elif content["type"] == "text":
                            text = content["text"]
                    elif isinstance(content, str):
                        text = content

    return image_url, text


def save_base64_image(base64_string, filename):
    # Split the base64 string header to get the actual data
    header, data = base64_string.split(",", 1)

    # Determine the file extension (e.g., '.png', '.jpg')
    ext = header.split("/")[1].split(";")[0]

    # Decode the base64 data
    binary_data = base64.b64decode(data)

    # Create a BytesIO object from the decoded data
    image_bytes = BytesIO(binary_data)

    # Open the image using PIL
    image = Image.open(image_bytes)

    # Save the image as a PNG file
    image.save(f"{filename}.{ext}", ext)

    return f"{filename}.{ext}"
