"""Convert a PaRoutes route-set-specific Keras expansion-policy model to ONNX
WITHOUT TensorFlow.

The PaRoutes models (`uspto_rxn_n{1,5}_keras_model.hdf5`, Keras 2.6) are simple
MLPs:  Input(2048) -> Dense(512, elu) -> Dropout(0.4) -> Dense(n_templates, softmax).
Dropout is the identity at inference, so we read the four weight arrays straight
out of the HDF5 with PyTables, rebuild the exact forward pass in PyTorch, and
export to ONNX with the same input/output signature AiZynthFinder's
LocalOnnxModel expects (input `dense_input` [batch,2048], output `dense_1`
[batch,n_templates], softmax probabilities).

Verified equivalence assumptions (checked against the stored model_config):
  layers = [InputLayer, Dense(elu), Dropout, Dense(softmax)]
  dense.kernel [2048,512] dense.bias [512]
  dense_1.kernel [512,N]  dense_1.bias [N]
Keras Dense computes  y = act(x @ kernel + bias); torch Linear weight = kernel.T.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import tables
import torch
import torch.nn as nn


def load_keras_weights(path: str):
    f = tables.open_file(path, "r")
    try:
        cfg = f.root._v_attrs["model_config"]
        cfg = json.loads(cfg.decode() if isinstance(cfg, bytes) else cfg)
        layers = [(l["class_name"], l.get("config", {})) for l in cfg["config"]["layers"]]
        # sanity: exactly the architecture we support
        kinds = [k for k, _ in layers]
        assert kinds == ["InputLayer", "Dense", "Dropout", "Dense"], f"unexpected arch: {kinds}"
        act1 = layers[1][1].get("activation")
        act2 = layers[3][1].get("activation")
        assert act1 == "elu" and act2 == "softmax", f"unexpected activations: {act1},{act2}"

        def arr(name):
            return np.asarray(f.get_node(name)[:])

        w0 = arr("/model_weights/dense/dense/kernel:0")      # [2048,512]
        b0 = arr("/model_weights/dense/dense/bias:0")        # [512]
        w1 = arr("/model_weights/dense_1/dense_1/kernel:0")  # [512,N]
        b1 = arr("/model_weights/dense_1/dense_1/bias:0")    # [N]
    finally:
        f.close()
    return w0, b0, w1, b1


class ExpansionMLP(nn.Module):
    def __init__(self, w0, b0, w1, b1):
        super().__init__()
        din, dh = w0.shape
        dh2, dout = w1.shape
        assert dh == dh2
        self.fc1 = nn.Linear(din, dh)
        self.fc2 = nn.Linear(dh, dout)
        self.act = nn.ELU()  # keras 'elu' == alpha=1.0, same as torch default
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(w0.T.copy()))
            self.fc1.bias.copy_(torch.from_numpy(b0.copy()))
            self.fc2.weight.copy_(torch.from_numpy(w1.T.copy()))
            self.fc2.bias.copy_(torch.from_numpy(b1.copy()))

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="keras hdf5 model")
    ap.add_argument("--out", required=True, help="output onnx path")
    args = ap.parse_args()

    w0, b0, w1, b1 = load_keras_weights(args.inp)
    din = w0.shape[0]
    dout = w1.shape[1]
    model = ExpansionMLP(w0, b0, w1, b1).eval()

    dummy = torch.zeros(1, din, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["dense_input"], output_names=["dense_1"],
        dynamic_axes={"dense_input": {0: "batch"}, "dense_1": {0: "batch"}},
        opset_version=13,
    )

    # verify with onnxruntime
    import onnxruntime as ort
    s = ort.InferenceSession(args.out)
    isp = s.get_inputs()[0]
    osp = s.get_outputs()[0]
    assert int(isp.shape[1]) == din, (isp.shape, din)
    assert int(osp.shape[1]) == dout, (osp.shape, dout)
    # numerical sanity on a few random binary fingerprints
    rng = np.random.RandomState(0)
    x = (rng.rand(4, din) > 0.97).astype(np.float32)
    y = s.run([osp.name], {isp.name: x})[0]
    torch_y = model(torch.from_numpy(x)).detach().numpy()
    max_abs = float(np.max(np.abs(y - torch_y)))
    print(f"[{args.out}] in={din} out={dout} rowsum={y.sum(1).round(4).tolist()} "
          f"nonneg={bool((y >= 0).all())} onnx_vs_torch_maxabs={max_abs:.2e}")
    assert max_abs < 1e-5 and bool((y >= 0).all())
    print("OK")


if __name__ == "__main__":
    main()
