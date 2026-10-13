"""One-off: build the model files config.yaml enables, into models/ (sha256 printed for pinning).

    uv run --group export python export.py

Classifiers are exported from Hugging Face to ONNX, then INT8-quantized (`<name>-int8.onnx`); NudeNet's
ONNX weights are downloaded from its GitHub release. PyTorch is only needed here, never in the service.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

import torch
import yaml

OUT = Path("models")
HF = {"freepik": "Freepik/nsfw_image_detector", "falconsai": "Falconsai/nsfw_image_detection"}
NUDENET = "https://api.github.com/repos/notAI-tech/NudeNet/releases/tags/v3.4-weights"  # browser URL needs login


class Logits(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).logits


def classifier(name: str, size: int, out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForImageClassification
    fp32 = OUT / f"{name}.onnx"
    model = Logits(AutoModelForImageClassification.from_pretrained(HF[name], dtype=torch.float32)).eval()
    torch.onnx.export(model, torch.randn(1, 3, size, size), str(fp32), opset_version=17, dynamo=False,
                      input_names=["pixels"], output_names=["logits"],
                      dynamic_axes={"pixels": {0: "n"}, "logits": {0: "n"}})
    quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8)
    fp32.unlink()


def nudenet(variant: str, out: Path) -> None:
    assets = json.load(urllib.request.urlopen(NUDENET))["assets"]
    url = next(a["url"] for a in assets if a["name"] == f"{variant}.onnx")
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    out.write_bytes(urllib.request.urlopen(req).read())


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, m in yaml.safe_load(open("config.yaml"))["models"].items():
        if not m.get("enabled") or m["kind"] == "quality":  # quality checks have no weights
            continue
        out = OUT / f"{m.get('file', name)}.onnx"
        if m["kind"] == "detector":
            nudenet(m["file"].removeprefix("nudenet-"), out)
        else:
            classifier(name, m["size"], out)
        print(f"{out}  {out.stat().st_size / 1e6:.0f} MB  sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
