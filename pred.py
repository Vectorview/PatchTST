from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, SparseTSF
import numpy as np
import pandas as pd
import argparse
import torch
import os

config = {
        "seq_len": 96,
        "pred_len": 96,
        "enc_in": 7, 
        "period_len": 12,
}
args = argparse.Namespace(**config)

model = SparseTSF.Model(args).float()

setting = "NewModel_period_12_epoch15_lr0.002_bs128_itr1_stride16_plen24_hdo0.1_do0.1_dff128_dm16_heads4_elay3_enc7/"
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
