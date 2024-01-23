from SPHINX import SPHINXModel
from PIL import Image
import torch

# Besides loading the `consolidated.*.pth` model weights, from_pretrained will also try to 
# use `tokenizer.model', 'meta.json', and 'config.json' under `pretrained_path` to configure
# the `tokenizer_path`, `llama_type`, and `llama_config` of the model. You may also override
# the configurations by explicitly specifying the arguments
model = SPHINXModel.from_pretrained(pretrained_path="/app/finetune/mm/SPHINX/SPHINX-Tiny", with_visual=True)

image = Image.open("examples/3.jpg")
qas = [["What's in the image?", None]]

response = model.generate_response(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)

print(response)

# if you wanna continue
qas[-1][-1] = response
qas.append(["Then how does it look like?", None])
response2 = model.generate_response(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)

print(response2)