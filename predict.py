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

from cog import BaseModel, BasePredictor
from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg

class Output(BaseModel):
    svg: str


class Predictor(BasePredictor):
    def setup(self):
        model_name = "starvector/starvector-8b-im2svg"
        self.model = StarVectorForCausalLM.from_pretrained(model_name)
        self.model.cuda()
        self.model.eval()

    def predict(
        self,
        image_path: str,
    ) -> Output:
        image_pil = Image.open(image_path)
        image = self.model.process_images([image_pil])[0].cuda()
        batch = {"image": image}
        raw_svg = self.model.generate_im2svg(batch, max_length=1000)[0]
        svg, raster_image = process_and_rasterize_svg(raw_svg)
        return Output(svg=svg)
