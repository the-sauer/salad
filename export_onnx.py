"""Export a DINOv2 SALAD checkpoint to ONNX.

The exported graph takes a normalized image batch [B, 3, H, W] (ImageNet mean/std,
H and W fixed at export time and divisible by 14) and returns L2-normalized global
descriptors [B, m*l + g]. The batch dimension is dynamic.

Example:
    python export_onnx.py --ckpt_path weights/dino_salad.ckpt --output weights/dino_salad.onnx
"""
import os
# xformers' memory_efficient_attention cannot be exported; fall back to plain attention
os.environ["XFORMERS_DISABLED"] = "1"

import argparse
import math
import types

import torch
import torch.nn as nn

from vpr_model import VPRModel
from models.aggregators.salad import SALAD


def onnx_matching_probs(S, dustbin_score, num_iters=3, reg=1.0):
    """Same as models.aggregators.salad.get_matching_probs, but without in-place
    writes and dim-less squeeze() so it traces correctly for any batch size."""
    B, m, n = S.shape
    S_aug = torch.cat([S, dustbin_score.expand(B, 1, n).to(S.dtype)], dim=1)

    norm = -math.log(n + m)
    log_a = torch.tensor([[norm] * m + [norm + math.log(n - m)]], dtype=S.dtype, device=S.device)
    log_b = torch.full((1, n), norm, dtype=S.dtype, device=S.device)

    M = S_aug / reg
    u = torch.zeros_like(M[:, :, 0])
    v = torch.zeros_like(M[:, 0, :])
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1)

    return M + u.unsqueeze(2) + v.unsqueeze(1) - norm


def salad_forward(self, x):
    """SALAD.forward using onnx_matching_probs and an einsum instead of the
    repeat/multiply/sum, which avoids materializing a [B, l, m, N] tensor."""
    x, t = x

    f = self.cluster_features(x).flatten(2)
    p = self.score(x).flatten(2)
    t = self.token_features(t)

    p = onnx_matching_probs(p, self.dust_bin, 3)
    p = torch.exp(p)[:, :-1, :]

    agg = torch.einsum("bln,bmn->blm", f, p)

    f = torch.cat([
        nn.functional.normalize(t, p=2, dim=-1),
        nn.functional.normalize(agg, p=2, dim=1).flatten(1)
    ], dim=-1)

    return nn.functional.normalize(f, p=2, dim=-1)


def load_model(ckpt_path, descriptor_size=None):
    if descriptor_size is None or descriptor_size == 256:
        agg_config = {
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        }
    elif descriptor_size == 64:
        agg_config = {
            'num_channels': 768,
            'num_clusters': 32,
            'cluster_dim': 64,
            'token_dim': 64,
        }
    elif descriptor_size == 32:
        agg_config = {
            'num_channels': 768,
            'num_clusters': 16,
            'cluster_dim': 32,
            'token_dim': 32,
        }
    else:
        raise ValueError

    model = VPRModel(
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config=agg_config,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model.eval()


def patch_for_export(model, image_size):
    """Make the model ONNX-friendly for a fixed input size."""
    dino = model.backbone.model
    h, w = image_size

    # Bicubic interpolation of the position embeddings is done once here in PyTorch;
    # ONNX's cubic Resize does not match torch numerically.
    dummy = torch.zeros(1, 1 + (h // 14) * (w // 14), dino.embed_dim)
    with torch.no_grad():
        pos_embed = dino.interpolate_pos_encoding(dummy, h, w).detach().clone()

    def fixed_pos_encoding(self, x, w_, h_):
        assert (w_, h_) == (h, w), f"model was exported for {image_size}, got {(w_, h_)}"
        return pos_embed

    dino.interpolate_pos_encoding = types.MethodType(fixed_pos_encoding, dino)

    assert isinstance(model.aggregator, SALAD)
    model.aggregator.forward = types.MethodType(salad_forward, model.aggregator)
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SALAD to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output .onnx path (default: next to ckpt)")
    parser.add_argument("--desc_size", type=int, default=None, help="Dimensionality of the descriptor")
    parser.add_argument("--image_size", nargs='+', type=int, default=[322, 322], help="Input size H W (divisible by 14)")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--no_verify", action="store_true", help="Skip onnxruntime comparison")

    args = parser.parse_args()
    if len(args.image_size) == 1:
        args.image_size = args.image_size * 2
    args.image_size = tuple(args.image_size)
    assert len(args.image_size) == 2 and all(s % 14 == 0 for s in args.image_size), \
        "image_size must be one or two ints divisible by 14"
    args.output = args.output or os.path.splitext(args.ckpt_path)[0] + ".onnx"
    return args


def verify(model, onnx_path, image_size):
    import numpy as np
    import onnxruntime as ort

    x = torch.randn(3, 3, *image_size)
    with torch.no_grad():
        ref = model(x).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": x.numpy()})[0]

    max_diff = np.abs(ref - out).max()
    cos = (ref * out).sum(-1).min()
    print(f"onnxruntime check: output {out.shape}, max abs diff {max_diff:.2e}, min cosine sim {cos:.6f}")
    if max_diff > 1e-3:
        raise RuntimeError("ONNX output does not match PyTorch")


if __name__ == '__main__':
    args = parse_args()

    model = patch_for_export(load_model(args.ckpt_path, descriptor_size=args.desc_size), args.image_size)
    dummy = torch.randn(2, 3, *args.image_size)

    torch.onnx.export(
        model,
        (dummy,),
        args.output,
        input_names=["image"],
        output_names=["descriptor"],
        dynamic_axes={"image": {0: "batch"}, "descriptor": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Exported {args.output}")

    if not args.no_verify:
        verify(model, args.output, args.image_size)
