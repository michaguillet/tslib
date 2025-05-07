#!/usr/bin/env python3
"""
evaluate_timesnet_forecast.py

Load a pre-trained TimesNet long-term-forecast model, run it on selected samples,
and plot history + true future + forecast with Plotly.
"""

import argparse
import numpy as np
import torch
import plotly.graph_objects as go
from models.TimesNet import Model

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate TimesNet long-term forecast"
    )
    p.add_argument('--data_path',   type=str, required=True,
                   help="Path to the .npy test data array")
    p.add_argument('--model_path',  type=str, required=True,
                   help="Path to the saved TimesNet checkpoint .pth")
    p.add_argument('--samples',     type=int, nargs='+', default=[0],
                   help="Indices of samples to plot (e.g. --samples 0 5 10)")
    p.add_argument('--seq_len',     type=int, default=96,
                   help="Encoder input length")
    p.add_argument('--label_len',   type=int, default=48,
                   help="Decoder “start” tokens length")
    p.add_argument('--pred_len',    type=int, default=96,
                   help="Number of steps to forecast")
    p.add_argument('--d_model',     type=int, default=16,
                   help="Model embedding dimension (must match training)") 
    p.add_argument('--num_kernels', type=int, default=5,
                   help="Number of inception kernels (must match training)")
    p.add_argument('--d_ff',        type=int, default=32,
                   help="Feed-forward hidden dimension (must match training)")
    p.add_argument('--e_layers',    type=int, default=2,
                   help="Number of encoder layers")
    p.add_argument('--dropout',     type=float, default=0.1,
                   help="Dropout probability (must match training)")
    p.add_argument('--freq',        type=str, default='h',
                   help="Temporal frequency string")
    p.add_argument('--use_gpu',     action='store_true',
                   help="Use CUDA if available")
    p.add_argument('--top_k',       type=int, default=3,
                   help="Number of top-k attention heads to use")
    return p.parse_args()

def main():
    args = parse_args()

    # ---- load & reshape data ----
    data = np.load(args.data_path)
    # data.shape == (num_samples, axes, markers, frames)
    num_samples, axes, markers, frames = data.shape
    print("Loaded data shape:", data.shape)

    # to [S, T, enc_in]
    x = torch.tensor(data, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2) \
         .reshape(num_samples, frames, axes * markers)
    x = x[args.samples]
    S, T, enc_in = x.shape
    print(f"Using {S} samples {args.samples}, input shape:", x.shape)

    # ---- build TimesNet config ----
    class Config: pass
    config = Config()
    config.task_name   = 'long_term_forecast'
    config.seq_len     = args.seq_len
    config.label_len   = args.label_len
    config.pred_len    = args.pred_len
    config.enc_in      = enc_in
    config.dec_in      = enc_in
    config.c_out       = enc_in
    config.d_model     = args.d_model
    config.num_kernels = args.num_kernels
    config.top_k       = args.top_k
    config.d_ff        = args.d_ff
    config.e_layers    = args.e_layers
    config.dropout     = args.dropout
    config.embed       = 'timeF'
    config.freq        = args.freq

    # ---- instantiate & load ----
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available()
                          else 'cpu')
    model = Model(config).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    sd    = ckpt.get('state_dict', ckpt)
    model.load_state_dict(sd, strict=False)  # skips predict_linear if present
    model.eval()
    print("Loaded checkpoint onto", device)

    # ---- forward ----
    x_dev = x.to(device)  # [S, T, enc_in]
    # split history vs future
    enc_input = x_dev[:, :args.seq_len, :]  # encoder sees first seq_len
    zero_fut  = torch.zeros((S, args.pred_len, enc_in), device=device)
    dec_input = torch.cat([
        enc_input[:, -args.label_len :, :],  # last label_len of history
        zero_fut
    ], dim=1)
    # no time‐marks => pass None for both mark args
    out = model(enc_input, None, dec_input, None)  # [S, seq_len?+pred_len, enc_in]
    # take only the forecast portion
    pred = out[:, -args.pred_len :, :].cpu().detach().numpy()
    # true future
    true = x_dev[:, args.seq_len:args.seq_len+args.pred_len, :].cpu().numpy()

    # ---- plot ----
    feat = 83  # choose channel to visualize
    fig = go.Figure()
    for i, idx in enumerate(args.samples):
        hist_x = np.arange(0, args.seq_len)
        fut_x  = np.arange(args.seq_len, args.seq_len + args.pred_len)

        # history
        fig.add_trace(go.Scatter(
            x=hist_x, y=x_dev[i, :args.seq_len, feat].cpu().numpy(),
            mode='lines', name=f'Sample{idx} history'
        ))
        # ground truth
        fig.add_trace(go.Scatter(
            x=fut_x, y=true[i,:,feat],
            mode='lines', name=f'Sample{idx} true'
        ))
        # forecast
        fig.add_trace(go.Scatter(
            x=fut_x, y=pred[i,:,feat],
            mode='lines', name=f'Sample{idx} pred'
        ))

    fig.update_layout(
        title="TimesNet Long-Term Forecast",
        xaxis_title="Time step",
        yaxis_title="Value",
    )
    fig.show()

if __name__ == '__main__':
    main()
