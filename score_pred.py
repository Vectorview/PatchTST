import numpy as np
import pandas as pd
from utils.metrics import metric

run = "PatchTST_period_4_epoch15_lr0.001_bs128_itr1_stride16_plen24_hdo0.1_do0.3_dff128_dm128_heads4_elay3_enc7"

preds = np.load(f"results/{run}/pred.npy")

data_list = []
for i in range(1, 11):
    file_path = f'dataset/test_data_y_{i}.csv'
    df = pd.read_csv(file_path)
    last_96_rows = df.iloc[-96:, 1:].to_numpy()
    data_list.append(last_96_rows)
trues = np.array(data_list)

mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
print(mse)
