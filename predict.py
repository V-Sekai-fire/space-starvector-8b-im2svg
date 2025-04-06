#!/usr/bin/env python3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cog import BaseModel, BasePredictor, Path, Input
from PIL import Image
from transformers import AutoModelForCausalLM
import torch

class Output(BaseModel):
    svg: str

class Predictor(BasePredictor):
    def setup(self):
        model_name = "starvector/starvector-8b-im2svg"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
        self.processor = self.model.model.processor
        self.tokenizer = self.model.model.svg_transformer.tokenizer

        self.model.cuda()
        self.model.eval()

    def predict(
        self,
        image_path: Path = Input(description="Path to the input image"),
    ) -> Output:
        image_pil = Image.open(image_path)
        image = self.processor(image_pil, return_tensors="pt")["pixel_values"].cuda()
        if not image.shape[0] == 1:
            image = image.squeeze(0)
        batch = {"image": image}

        raw_svg = self.model.generate_im2svg(batch, max_length=4000)[0]
        return Output(svg=raw_svg)
