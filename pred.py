from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, SparseTSF
import numpy as np
import pandas as pd
import argparse
import torch
import os

from models import PatchTST
config = {
        "seq_len": 96,
        "pred_len": 96,
        "enc_in": 7, 
        "period_len": 12,
        "d_model": 128,
        "e_layers": 3,
        "n_heads": 4,
        "d_ff": 128,
        "dropout": 0.1,
        "fc_dropout": 0.3,
        "head_dropout": 0.1,
        "individual": 0,
        "patch_len": 24,
        "stride": 16,
        "padding_patch": "end",
        "revin": 1,
        "affine": 0,
        "subtract_last": 0,
        "decomposition": 0,
        "kernel_size": 25,
}
args = argparse.Namespace(**config)

model = PatchTST.Model(args).float()

setting = "PatchTST_period_4_epoch5_lr0.001_bs128_itr1_stride16_plen24_hdo0.1_do0.3_dff128_dm128_heads4_elay3_enc7"
path = f"checkpoints/{setting}/checkpoint.pth"
model.load_state_dict(torch.load(path))
model.eval()

preds = []
with torch.no_grad():
    for i in range(1, 11):
        print(i)
        df_raw = pd.read_csv(f"dataset/test_data_x_{i}.csv")
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        batch_x = torch.tensor(df_data.values).float().to("cpu").reshape(1, 96, 7)
        outputs = model(batch_x) # shape: 1, 96, 7
        preds.append(outputs)
preds = np.array(preds) # shape: 10, 1, 96, 7
preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1]) # shape: 10, 96, 7
