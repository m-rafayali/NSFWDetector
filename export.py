"""One-off: download pretrained models and export them to models/<name>.onnx (+ INT8 variants).

    uv run --group export python export.py [name ...]

PyTorch is only needed here; the service runs on onnxruntime alone.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

import torch

OUT = Path("models")
NUDENET = "https://api.github.com/repos/notAI-tech/NudeNet/releases/tags/v3.4-weights"  # browser URL now needs login
ADAMCODD = "https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/onnx/model.onnx"


class Logits(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return getattr(out, "logits", out)


def to_onnx(model, size, path):
    model = Logits(model).eval().float()
    torch.onnx.export(model, torch.randn(1, 3, size, size), str(path), opset_version=17, dynamo=False,
                      input_names=["pixels"], output_names=["logits"],
                      dynamic_axes={"pixels": {0: "n"}, "logits": {0: "n"}})


def hf_classifier(repo):
    from transformers import AutoModelForImageClassification
    return AutoModelForImageClassification.from_pretrained(repo, torch_dtype=torch.float32)


def export(name):
    path = OUT / f"{name}.onnx"
    if name == "freepik":
        to_onnx(hf_classifier("Freepik/nsfw_image_detector"), 448, path)
    elif name == "falconsai":
        to_onnx(hf_classifier("Falconsai/nsfw_image_detection"), 224, path)
    elif name == "marqo":
        import timm
        to_onnx(timm.create_model("hf_hub:Marqo/nsfw-image-detection-384", pretrained=True), 384, path)
    elif name == "adamcodd":
        urllib.request.urlretrieve(ADAMCODD, path)
    elif name.startswith("nudenet-"):
        import json
        assets = json.load(urllib.request.urlopen(NUDENET))["assets"]
        url = next(a["url"] for a in assets if a["name"] == name.split("-")[1] + ".onnx")
        req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
        path.write_bytes(urllib.request.urlopen(req).read())
        return [path]
    # INT8 dynamic quantization variant for the speed/accuracy bake-off.
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = OUT / f"{name}-int8.onnx"
    quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
    return [path, q]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    names = sys.argv[1:] or ["freepik", "marqo", "falconsai", "adamcodd", "nudenet-320n", "nudenet-640m"]
    for n in names:
        for p in export(n):
            print(f"{p}  {p.stat().st_size / 1e6:.0f} MB  sha256={hashlib.sha256(p.read_bytes()).hexdigest()}")
