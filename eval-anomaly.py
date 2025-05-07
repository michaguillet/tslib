#!/usr/bin/env python3
"""
evaluate_timesnet_anomaly.py

Load a pre-trained TimesNet anomaly detection model, run it on selected windows,
and plot the original vs. reconstruction and anomaly scores with Plotly.
Supports both 3D window arrays ([N, T, enc_in]) and 4D raw arrays
([N, axes, markers, frames]) by reshaping appropriately.
"""
import argparse
import numpy as np
import torch
import plotly.graph_objects as go
from models.TimesNet import Model


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate TimesNet Anomaly Detection"
    )
    p.add_argument('--data_path',  type=str, required=True,
                   help="Path to the .npy test windows or raw array file")
    p.add_argument('--model_path', type=str, required=True,
                   help="Path to the saved TimesNet checkpoint .pth")
    p.add_argument('--samples',    type=int, nargs='+', default=[0],
                   help="Indices of windows to plot (e.g. --samples 0 5 10)")
    p.add_argument('--d_model',    type=int, default=64,
                   help="Model embedding dimension (must match training)")
    p.add_argument('--num_kernels',type=int, default=5,
                   help="Number of inception kernels (must match training)")
    p.add_argument('--d_ff',       type=int, default=64,
                   help="Feed-forward hidden dimension (must match training)")
    p.add_argument('--e_layers',   type=int, default=2,
                   help="Number of encoder layers (must match training)")
    p.add_argument('--dropout',    type=float, default=0.1,
                   help="Dropout probability (must match training)")
    p.add_argument('--freq',       type=str, default='h',
                   help="Temporal frequency string (unused in anomaly)")
    p.add_argument('--use_gpu',    action='store_true',
                   help="Use CUDA if available")
    p.add_argument('--top_k',      type=int, default=5,
                   help="Number of top-k attention heads (unused in anomaly)")
    p.add_argument('--feat',       type=int, default=0,
                   help="Feature/channel index to plot (default 0)")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # ---- load data ----
    raw = np.load(args.data_path)
    # handle both 3D windows [N, T, enc_in] and 4D raw [N, axes, markers, frames]
    if raw.ndim == 3:
        data = raw
    elif raw.ndim == 4:
        N, axes, markers, frames = raw.shape
        data = raw.transpose(0, 3, 1, 2).reshape(N, frames, axes * markers)
    else:
        raise ValueError(f"Unsupported array shape {raw.shape}; need 3D or 4D .npy")

    num_windows, T, enc_in = data.shape
    print(f"Loaded data shape: {raw.shape} -> windows {num_windows}, length {T}, features {enc_in}")

    # select windows
    x = torch.tensor(data[args.samples], dtype=torch.float32)  # [S, T, enc_in]
    S = x.shape[0]

    # ---- build TimesNet config ----
    class Config: pass
    config = Config()
    config.task_name   = 'anomaly_detection'
    config.seq_len     = T  # must match actual window length
    config.label_len   = 0
    config.pred_len    = 0
    config.enc_in      = enc_in
    config.dec_in      = enc_in
    config.c_out       = enc_in
    config.d_model     = args.d_model
    config.num_kernels = args.num_kernels
    config.top_k       = args.top_k
    config.d_ff        = args.d_ff
    config.e_layers    = args.e_layers
    config.dropout     = args.dropout
    config.embed       = 'fixed'
    config.freq        = args.freq

    # ---- instantiate & load ----
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    model = Model(config).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    sd    = ckpt.get('state_dict', ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("Loaded checkpoint onto", device)

    # ---- forward ----
    x_dev = x.to(device)  # [S, T, enc_in]
    with torch.no_grad():
        out = model(x_dev, None, None, None)  # [S, T, enc_in]
    out_np = out.cpu().numpy()
    x_np   = x_dev.cpu().numpy()

    # per-time-step MSE
    mse_ts = np.mean((x_np - out_np) ** 2, axis=2)  # [S, T]

    # ---- plot ----
    timesteps = np.arange(T)
    fig = go.Figure()
    for i, idx in enumerate(args.samples):
        # original vs reconstruction for chosen feature
        fig.add_trace(go.Scatter(
            x=timesteps, y=x_np[i,:,args.feat],
            mode='lines', name=f'Sample{idx} Original feat{args.feat}'
        ))
        fig.add_trace(go.Scatter(
            x=timesteps, y=out_np[i,:,args.feat],
            mode='lines', name=f'Sample{idx} Recon feat{args.feat}'
        ))
        # anomaly score
        fig.add_trace(go.Scatter(
            x=timesteps, y=mse_ts[i],
            mode='lines', name=f'Sample{idx} MSE'
        ))

    fig.update_layout(
        title="TimesNet Anomaly Detection",
        xaxis_title="Time step",
        yaxis_title="Value / MSE",
    )
    fig.show()
