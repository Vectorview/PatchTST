if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
pred_len=96
label_len=12

model_name=NewModel

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021

# PatchTST_period_12_epoch15_lr0.02_bs128_itr1_stride8_plen72_hdo0.2_do0.3_dff128_dm16_heads4_elay3_enc7  
# mse:0.38578662276268005, mae:0.435549259185791, rse:0.5805014967918396

# PatchTST_period_12_epoch15_lr0.02_bs64_itr1_stride8_plen48_hdo0.2_do0.3_dff128_dm16_heads4_elay3_enc7  
# mse:0.3831917941570282, mae:0.4329233765602112, rse:0.5785459280014038
#
# PatchTST_period_12_epoch15_lr0.02_bs64_itr1_stride8_plen24_hdo0.1_do0.1_dff64_dm16_heads4_elay3_enc7
#  0.3411906


# NewModel_period_12_epoch15_lr0.02_bs128_itr1_stride16_plen48_hdo0.1_do0.1_dff128_dm16_heads4_elay3_enc7
#  0.37132364
#  NewModel_period_12_epoch15_lr0.05_bs128_itr1_stride16_plen24_hdo0.1_do0.1_dff128_dm32_heads4_elay3_enc7
#  0.37059333
#  NewModel_period_6_epoch15_lr0.08_bs128_itr1_stride16_plen24_hdo0.1_do0.1_dff128_dm32_heads4_elay3_enc7
#  0.36746519
#  NewModel_period_4_epoch15_lr0.1_bs128_itr1_stride16_plen24_hdo0.1_do0.3_dff128_dm32_heads4_elay3_enc7
#  0.36366724
#  NewModel_period_4_epoch25_lr0.12_bs128_itr1_stride16_plen24_hdo0.1_do0.1_dff128_dm32_heads4_elay3_enc7
#  0.35877394




for train_epochs in 25; do
  for learning_rate in 0.12; do
    for batch_size in 128; do
      for itr in 1; do
        for stride in 16; do
          for patch_len in 24; do
            for head_dropout in 0.1; do
              for fc_dropout in 0.1; do
                for dropout in 0.1; do
                  for d_ff in 128; do
                    for d_model in 32; do
                      for n_heads in 4; do
                        for e_layers in 3; do
                          for enc_in in 7; do
                            for period_len in 4; do
                              python -u run_longExp.py \
                                --random_seed $random_seed \
                                --is_training 1 \
                                --root_path $root_path_name \
                                --data_path $data_path_name \
                                --model_id $model_id_name_$seq_len'_'$pred_len \
                                --model $model_name \
                                --data $data_name \
                                --features M \
                                --seq_len $seq_len \
                                --pred_len $pred_len \
                                --enc_in $enc_in \
                                --e_layers $e_layers \
                                --n_heads $n_heads \
                                --d_model $d_model \
                                --d_ff $d_ff \
                                --dropout $dropout \
                                --fc_dropout $fc_dropout \
                                --head_dropout $head_dropout \
                                --patch_len $patch_len \
                                --stride $stride \
                                --des 'Exp' \
                                --train_epochs $train_epochs \
                                --period_len $period_len \
                                --patience 5 \
                                --itr $itr --batch_size $batch_size --learning_rate $learning_rate
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
