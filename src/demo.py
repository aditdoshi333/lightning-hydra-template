import pyrootutils
import torchvision.transforms as T

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
import os

# SETTING ENV VARIABLE FOR GRADIO SERVER PORT

os.environ["GRADIO_SERVER_PORT"] = "8090"


def get_cifar_class_label(idx):
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ][idx]

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    # log.info("Running Demo")

    # log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    # log.info(f"Loaded Model: {model}")
    

    def cifar_inference(image):
        if image is None:
            return None
        image = torch.tensor(image[None,...], dtype=torch.float32)
        image = image.permute(0, 3, 1, 2)        
        preds = model.forward_jit(image)
        print(preds)
        preds = preds[0].tolist()
        return {get_cifar_class_label(i): preds[i] for i in range(10)}

    im = gr.Image(type="numpy")

    demo = gr.Interface(
        fn=cifar_inference,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(server_port=8090, server_name="0.0.0.0")

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()