#!/bin/bash

current_time=$(date +"%m%d_%H%M")

seq_len_meta=96
testname=MFIKAN_experiment
pred_lens_meta=(96 192 336 720)
epochs=(20 20 20 20)
label_len_meta=48
lradj='type3'

declare -A batch_sizes_per_dataset
batch_sizes_per_dataset["ETTh1"]=128
batch_sizes_per_dataset["ETTh2"]=32
batch_sizes_per_dataset["ETTm1"]=256
batch_sizes_per_dataset["ETTm2"]=256
batch_sizes_per_dataset["weather"]=64
batch_sizes_per_dataset["traffic"]=8
batch_sizes_per_dataset["electricity"]=8
batch_sizes_per_dataset["exchange_rate"]=16
batch_sizes_per_dataset["national_illness"]=32
batch_sizes_per_dataset["solar"]=8
batch_sizes_per_dataset["PEMS03"]=8
batch_sizes_per_dataset["PEMS04"]=8
batch_sizes_per_dataset["PEMS07"]=8
batch_sizes_per_dataset["PEMS08"]=8


declare -A learning_rates_per_dataset
learning_rates_per_dataset["ETTh1"]=0.0005
learning_rates_per_dataset["ETTh2"]=0.0001
learning_rates_per_dataset["ETTm1"]=0.0001
learning_rates_per_dataset["ETTm2"]=0.0005
learning_rates_per_dataset["weather"]=0.0001
learning_rates_per_dataset["traffic"]=0.0001
learning_rates_per_dataset["electricity"]=0.0001
learning_rates_per_dataset["exchange_rate"]=0.0001
learning_rates_per_dataset["national_illness"]=0.001
learning_rates_per_dataset["solar"]=0.0005
learning_rates_per_dataset["PEMS03"]=0.0003
learning_rates_per_dataset["PEMS04"]=0.0003
learning_rates_per_dataset["PEMS07"]=0.0003
learning_rates_per_dataset["PEMS08"]=0.0003

declare -A enc_in_per_dataset
enc_in_per_dataset["ETTh1"]=7
enc_in_per_dataset["ETTh2"]=7  
enc_in_per_dataset["ETTm1"]=7
enc_in_per_dataset["ETTm2"]=7 
enc_in_per_dataset["weather"]=21
enc_in_per_dataset["traffic"]=862
enc_in_per_dataset["electricity"]=321
enc_in_per_dataset["exchange_rate"]=8
enc_in_per_dataset["national_illness"]=7
enc_in_per_dataset["solar"]=137
enc_in_per_dataset["PEMS03"]=358
enc_in_per_dataset["PEMS04"]=307
enc_in_per_dataset["PEMS07"]=883
enc_in_per_dataset["PEMS08"]=170

declare -A data_per_dataset
data_per_dataset["ETTh1"]=ETTh1
data_per_dataset["ETTh2"]=ETTh2
data_per_dataset["ETTm1"]=ETTm1
data_per_dataset["ETTm2"]=ETTm2
data_per_dataset["weather"]=custom
data_per_dataset["traffic"]=custom
data_per_dataset["electricity"]=custom
data_per_dataset["exchange_rate"]=custom
data_per_dataset["national_illness"]=custom
data_per_dataset["solar"]=Solar
data_per_dataset["PEMS03"]=PEMS
data_per_dataset["PEMS04"]=PEMS
data_per_dataset["PEMS07"]=PEMS
data_per_dataset["PEMS08"]=PEMS

declare -A data_path_per_dataset
data_path_per_dataset["ETTh1"]=ETTh1.csv
data_path_per_dataset["ETTh2"]=ETTh2.csv
data_path_per_dataset["ETTm1"]=ETTm1.csv
data_path_per_dataset["ETTm2"]=ETTm2.csv
data_path_per_dataset["weather"]=weather.csv
data_path_per_dataset["traffic"]=traffic.csv
data_path_per_dataset["electricity"]=electricity.csv
data_path_per_dataset["exchange_rate"]=exchange_rate.csv
data_path_per_dataset["national_illness"]=national_illness.csv
data_path_per_dataset["solar"]=solar.txt
data_path_per_dataset["PEMS03"]=PEMS03.npz
data_path_per_dataset["PEMS04"]=PEMS04.npz
data_path_per_dataset["PEMS07"]=PEMS07.npz
data_path_per_dataset["PEMS08"]=PEMS08.npz

declare -A d_model_per_dataset
d_model_per_dataset["ETTh1"]=128
d_model_per_dataset["ETTh2"]=128
d_model_per_dataset["ETTm1"]=128
d_model_per_dataset["ETTm2"]=128
d_model_per_dataset["weather"]=256
d_model_per_dataset["traffic"]=128
d_model_per_dataset["electricity"]=128
d_model_per_dataset["exchange_rate"]=128
d_model_per_dataset["national_illness"]=256
d_model_per_dataset["solar"]=256
d_model_per_dataset["PEMS03"]=128
d_model_per_dataset["PEMS04"]=128
d_model_per_dataset["PEMS07"]=128
d_model_per_dataset["PEMS08"]=128


datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather" "exchange_rate" "national_illness" "electricity" "solar" "traffic")
model_name="MFIKAN"

for dataset in "${datasets[@]}"; do
    batch_size=${batch_sizes_per_dataset[$dataset]}
    learning_rate=${learning_rates_per_dataset[$dataset]}
    enc_in=${enc_in_per_dataset[$dataset]}
    data=${data_per_dataset[$dataset]}
    data_path=${data_path_per_dataset[$dataset]}
    d_model=${d_model_per_dataset[$dataset]}
    min_selected_features=${min_selected_features_per_dataset[$dataset]}
    max_selected_features=${max_selected_features_per_dataset[$dataset]}
    
    dec_in=$enc_in
    c_out=$enc_in
    c_in=$enc_in

    if [ "$dataset" = "national_illness" ]; then
        seq_len=36
        pred_lens=(24 36 48 60)
        label_len=18
    elif [[ "$dataset" =~ ^PEMS ]]; then
        seq_len=$seq_len_meta
        pred_lens=(12 24 48 96)
        label_len=$label_len_meta
    else
        seq_len=$seq_len_meta
        pred_lens=("${pred_lens_meta[@]}")
        label_len=$label_len_meta
    fi

    log_dir="./logs/$model_name/$testname/$dataset"
    mkdir -p "$log_dir"

    for i in "${!pred_lens[@]}"; do
        pred_len=${pred_lens[$i]}
        train_epoch=${epochs[$i]}
        log_file="$log_dir/${model_name}_${dataset}_${seq_len}_${pred_len}_${current_time}.log"

        echo "Running: $model_name on $dataset | lr=$learning_rate, bs=$batch_size, pred_len=$pred_len, min_feat=$min_selected_features, max_feat=$max_selected_features"
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path "$data_path" \
            --model_id "${dataset}_${seq_len}_${pred_len}" \
            --model "$model_name" \
            --data "$data" \
            --features M \
            --seq_len "$seq_len" \
            --label_len "$label_len" \
            --pred_len "$pred_len" \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in "$enc_in" \
            --dec_in "$dec_in" \
            --c_in "$c_in" \
            --c_out "$c_out" \
            --des 'Exp' \
            --itr 1 \
            --revin 1 \
            --d_model "$d_model" \
            --batch_size "$batch_size" \
            --learning_rate "$learning_rate" \
            --lradj "$lradj" \
            --train_epochs "$train_epoch" \
            --dropout 0.1 \
            --vmd_scales 3 \
            --vmd_alpha 1000 \
            --vmd_tau 0.2 \
            --vmd_dc 0 \
            --vmd_tol 1e-8 \
            --vmd_n_iter 500 \
            --vmd_init uniform \
            --vmd_chunk_size 8 \
            --kan_grid_size 8 \
            --kan_spline_order 3 \
            --kan_base_activation gelu \
            --kan_enable_bias \
            --kan_use_residual \
            --kan_use_layer_norm \
            --kan_hidden_dim 128 \
            --kan_dropout 0.3 \
            --use_dynamic_pruning \
            --dst_initial_sparsity 0.0 \
            --dst_target_sparsity 0.3 \
            --dst_schedule cosine \
            --dst_update_frequency 500 \
            --dst_total_steps 2000 \
            --num_workers 4 \
            --patience 10 \
            --use_amp > "$log_file" 2>&1
    done
done

echo "All training tasks are completed. Generating the results table..."
log_base_dir="./logs/$model_name/$testname"
output_csv_path="$log_base_dir/${model_name}_results.csv"

python generate_table.py --log_dir "$log_base_dir" --output_csv_path "$output_csv_path"
echo "Results table has been generated and saved."