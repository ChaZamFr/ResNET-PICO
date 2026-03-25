import argparse
import textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


INPUT_C     = 1
NUM_CLASSES = 10
L0_OUT_C    = 16
L1_C        = 16
L2_IN_C     = 16
L2_OUT_C    = 32
L3_C        = 32

EPOCHS      = 15
BATCH_SIZE  = 128
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


class ResBlock(nn.Module):
    """Standard residual block (same spatial size)."""
    def __init__(self, channels):
        super().__init__()
        self.conv_a = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_a   = nn.BatchNorm2d(channels)
        self.conv_b = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_b   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn_a(self.conv_a(x)))
        out = self.bn_b(self.conv_b(out))
        return F.relu(out + x)


class ResBlockDS(nn.Module):
    """Residual block with stride-2 downsampling + projection shortcut."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_a = nn.Conv2d(in_c,  out_c, 3, stride=2, padding=1, bias=False)
        self.bn_a   = nn.BatchNorm2d(out_c)
        self.conv_b = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn_b   = nn.BatchNorm2d(out_c)
        self.proj   = nn.Conv2d(in_c,  out_c, 1, stride=2, bias=False)
        self.bn_p   = nn.BatchNorm2d(out_c)

    def forward(self, x):
        shortcut = self.bn_p(self.proj(x))
        out = F.relu(self.bn_a(self.conv_a(x)))
        out = self.bn_b(self.conv_b(out))
        return F.relu(out + shortcut)


class PicoResNet(nn.Module):
    """
    ResNet-8 for 32x32 greyscale input, 10 classes.
    Matches resnet_pico.c layer-for-layer.
    """
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(INPUT_C, L0_OUT_C, 3, padding=1, bias=True)
        self.bn0   = nn.BatchNorm2d(L0_OUT_C)
        self.rb1   = ResBlock(L1_C)
        self.rb2   = ResBlockDS(L2_IN_C, L2_OUT_C)
        self.rb3   = ResBlock(L3_C)
        self.fc    = nn.Linear(L3_C, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))  # 32x32x16
        x = self.rb1(x)                       # 32x32x16
        x = self.rb2(x)                       # 16x16x32
        x = self.rb3(x)                       # 16x16x32
        x = x.mean(dim=[2, 3])                # global avg pool → 32
        return self.fc(x)


def get_dataloaders():
    tf = transforms.Compose([
        transforms.Pad(2),                       # 28x28 → 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_dl, test_dl



def train(model, train_dl, test_dl):
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out  = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            n          += len(y)
        sched.step()
        acc = evaluate(model, test_dl)
        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={total_loss/n:.4f}  "
              f"train_acc={correct/n*100:.2f}%  "
              f"test_acc={acc*100:.2f}%")


def evaluate(model, dl):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            n       += len(y)
    return correct / n


def fold_bn(conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, eps=1e-5):
    """Absorb BatchNorm parameters into conv weight and bias."""
    std    = np.sqrt(bn_var + eps)
    scale  = bn_gamma / std
    w_fold = conv_w * scale[:, None, None, None]
    b_fold = (conv_b - bn_mean) * scale + bn_beta
    return w_fold, b_fold

def np_(t):
    return t.detach().cpu().numpy().astype(np.float32)


def extract_weights(model):
    """
    Return a dict of float32 numpy arrays, BN-folded, in C layout:
    convolution weights are transposed to [kH][kW][in_c][out_c].
    """
    m = model.cpu().eval()
    w = {}

    # conv0 + bn0
    cw = np_(m.conv0.weight)   # [16, 1, 3, 3]
    cb = np_(m.conv0.bias)
    cw, cb = fold_bn(cw, cb,
                     np_(m.bn0.weight), np_(m.bn0.bias),
                     np_(m.bn0.running_mean), np_(m.bn0.running_var))
    w["w_conv0"] = cw.transpose(2, 3, 1, 0)   # → [3,3,1,16]
    w["b_conv0"] = cb

    def fold_rb(rb, prefix):
        wa = np_(rb.conv_a.weight)
        ba = np.zeros(wa.shape[0], np.float32)
        wb = np_(rb.conv_b.weight)
        bb = np.zeros(wb.shape[0], np.float32)
        wa, ba = fold_bn(wa, ba, np_(rb.bn_a.weight), np_(rb.bn_a.bias),
                         np_(rb.bn_a.running_mean), np_(rb.bn_a.running_var))
        wb, bb = fold_bn(wb, bb, np_(rb.bn_b.weight), np_(rb.bn_b.bias),
                         np_(rb.bn_b.running_mean), np_(rb.bn_b.running_var))
        w[f"w_{prefix}_a"] = wa.transpose(2, 3, 1, 0)
        w[f"b_{prefix}_a"] = ba
        w[f"w_{prefix}_b"] = wb.transpose(2, 3, 1, 0)
        w[f"b_{prefix}_b"] = bb

    fold_rb(m.rb1, "rb1")
    fold_rb(m.rb3, "rb3")

    # rb2 (downsampling + projection)
    rb2 = m.rb2
    wa = np_(rb2.conv_a.weight);  ba = np.zeros(wa.shape[0], np.float32)
    wb = np_(rb2.conv_b.weight);  bb = np.zeros(wb.shape[0], np.float32)
    wp = np_(rb2.proj.weight);    bp = np.zeros(wp.shape[0], np.float32)
    wa, ba = fold_bn(wa, ba, np_(rb2.bn_a.weight), np_(rb2.bn_a.bias),
                     np_(rb2.bn_a.running_mean), np_(rb2.bn_a.running_var))
    wb, bb = fold_bn(wb, bb, np_(rb2.bn_b.weight), np_(rb2.bn_b.bias),
                     np_(rb2.bn_b.running_mean), np_(rb2.bn_b.running_var))
    wp, bp = fold_bn(wp, bp, np_(rb2.bn_p.weight), np_(rb2.bn_p.bias),
                     np_(rb2.bn_p.running_mean), np_(rb2.bn_p.running_var))
    w["w_rb2_a"]    = wa.transpose(2, 3, 1, 0)   # [3,3,16,32]
    w["b_rb2_a"]    = ba
    w["w_rb2_b"]    = wb.transpose(2, 3, 1, 0)   # [3,3,32,32]
    w["b_rb2_b"]    = bb
    w["w_rb2_proj"] = wp.transpose(2, 3, 1, 0)   # [1,1,16,32]
    w["b_rb2_proj"] = bp

    # FC
    w["w_fc"] = np_(m.fc.weight).T   # [32, 10]
    w["b_fc"] = np_(m.fc.bias)       # [10]

    return w

def quantise(arr: np.ndarray) -> np.ndarray:
    q = np.round(arr * 256).astype(np.int64)
    n_clip = np.sum(np.abs(q) > 32767)
    if n_clip:
        print(f"  WARNING: {n_clip} values clipped to int16 range")
    return np.clip(q, -32768, 32767).astype(np.int16)

def array_to_c(name: str, arr: np.ndarray, cols: int = 16) -> str:
    flat = arr.flatten()
    rows = []
    for i in range(0, len(flat), cols):
        rows.append("    " + ", ".join(str(int(v)) for v in flat[i:i+cols]) + ",")
    dims = "".join(f"[{d}]" for d in arr.shape)
    return f"static const int16_t {name}{dims} = {{\n" + "\n".join(rows) + "\n};\n"


KEY_ORDER = [
    "w_conv0", "b_conv0",
    "w_rb1_a", "b_rb1_a", "w_rb1_b", "b_rb1_b",
    "w_rb2_a", "b_rb2_a", "w_rb2_b", "b_rb2_b",
    "w_rb2_proj", "b_rb2_proj",
    "w_rb3_a", "b_rb3_a", "w_rb3_b", "b_rb3_b",
    "w_fc",   "b_fc",
]


def generate_header(weights: dict) -> str:
    parts = [textwrap.dedent("""\
        /*
         * weights_mnist.h  — auto-generated by train_and_export.py
         * DO NOT EDIT MANUALLY.
         *
         * All values are Q8.8 fixed-point (int16_t).
         *
         * Usage in resnet_pico.c:
         *   1. Delete the zeroed placeholder arrays for every weight listed below.
         *   2. Add at the top of resnet_pico.c (before the weight section):
         *        #include "weights_mnist.h"
         *   3. Rebuild with cmake.
         */
        #pragma once
        #include <stdint.h>

    """)]

    for key in KEY_ORDER:
        arr = weights[key]
        q   = quantise(arr)
        parts.append(f"/* {key}  shape={arr.shape}  "
                     f"min={arr.min():.4f}  max={arr.max():.4f} */")
        parts.append(array_to_c(key, q))

    return "\n".join(parts)


def sanity_check(model, test_dl):
    model.eval()
    x, y = next(iter(test_dl))
    x = x.to(DEVICE)          # <-- move input to same device as model
    with torch.no_grad():
        preds = model(x[:8]).argmax(1).tolist()
    print("\nSanity check — first 8 test images:")
    print(f"  Predicted : {preds}")
    print(f"  True      : {y[:8].tolist()}")



def main():
    parser = argparse.ArgumentParser(description="Train ResNet on MNIST and export weights for Pico")
    parser.add_argument("--export-only", metavar="PTH",
                        help="Load an existing .pth file and skip training")
    parser.add_argument("--out",      default="weights_mnist.h",
                        help="Output header file (default: weights_mnist.h)")
    parser.add_argument("--save-pth", default="resnet_mnist.pth",
                        help="Where to save trained weights (default: resnet_mnist.pth)")
    args = parser.parse_args()

    model = PicoResNet().to(DEVICE)
    train_dl, test_dl = get_dataloaders()

    if args.export_only:
        print(f"Loading weights from {args.export_only} ...")
        model.load_state_dict(torch.load(args.export_only, map_location=DEVICE))
    else:
        print(f"Training on {DEVICE} for {EPOCHS} epochs ...")
        train(model, train_dl, test_dl)
        torch.save(model.state_dict(), args.save_pth)
        print(f"Weights saved → {args.save_pth}")

    acc = evaluate(model, test_dl)
    print(f"\nFinal test accuracy: {acc*100:.2f}%")

    sanity_check(model, test_dl)

    print("\nExtracting + quantising weights ...")
    weights = extract_weights(model)
    header  = generate_header(weights)

    with open(args.out, "w") as f:
        f.write(header)
    print(f"Header written → {args.out}")

    print(textwrap.dedent(f"""
    1. Copy weights_mnist.h into your Pico project folder.
    2. In resnet_pico.c, remove the zeroed weight arrays and add:
         #include "weights_mnist.h"
    3. Rebuild:
         cmake --build build
    4. Flash resnet_pico.uf2 to your Pico.
    """))


if __name__ == "__main__":
    main()
