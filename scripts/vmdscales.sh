#!/bin/bash

current_time=$(date +"%m%d_%H%M")

seq_len_meta=96
testname=mvmd_kan_experiment
pred_lens_meta=(96 192 336 720)
epochs=(30 30 30 30)
label_len_meta=48
lradj='type3'

# vmd_scales参数列表
vmd_scales_list=(2 3 4 5)

# === 定义每个数据集的专属参数（使用关联数组）===
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

declare -A min_selected_features_per_dataset
min_selected_features_per_dataset["ETTh1"]=2
min_selected_features_per_dataset["ETTh2"]=2
min_selected_features_per_dataset["ETTm1"]=2
min_selected_features_per_dataset["ETTm2"]=2
min_selected_features_per_dataset["weather"]=6
min_selected_features_per_dataset["traffic"]=258
min_selected_features_per_dataset["electricity"]=96
min_selected_features_per_dataset["exchange_rate"]=3
min_selected_features_per_dataset["national_illness"]=2
min_selected_features_per_dataset["solar"]=41
min_selected_features_per_dataset["PEMS03"]=41
min_selected_features_per_dataset["PEMS04"]=41
min_selected_features_per_dataset["PEMS07"]=88
min_selected_features_per_dataset["PEMS08"]=51

declare -A max_selected_features_per_dataset
max_selected_features_per_dataset["ETTh1"]=7
max_selected_features_per_dataset["ETTh2"]=7
max_selected_features_per_dataset["ETTm1"]=7
max_selected_features_per_dataset["ETTm2"]=7
max_selected_features_per_dataset["weather"]=21
max_selected_features_per_dataset["traffic"]=862
max_selected_features_per_dataset["electricity"]=321
max_selected_features_per_dataset["exchange_rate"]=8
max_selected_features_per_dataset["national_illness"]=7
max_selected_features_per_dataset["solar"]=137
max_selected_features_per_dataset["PEMS03"]=358
max_selected_features_per_dataset["PEMS04"]=307
max_selected_features_per_dataset["PEMS07"]=883
max_selected_features_per_dataset["PEMS08"]=170

datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather")
model_name="VMD_KAN"

# === 主循环 ===
for vmd_scales in "${vmd_scales_list[@]}"; do
    echo "=========================================================="
    echo "Running experiments with vmd_scales = $vmd_scales"
    echo "=========================================================="
    
    # 为每个vmd_scales值创建子目录
    testname_with_vmd="${testname}_vmd${vmd_scales}"
    
    for dataset in "${datasets[@]}"; do
        batch_size=${batch_sizes_per_dataset[$dataset]}
        learning_rate=${learning_rates_per_dataset[$dataset]}
        enc_in=${enc_in_per_dataset[$dataset]}
        data=${data_per_dataset[$dataset]}
        data_path=${data_path_per_dataset[$dataset]}
        d_model=${d_model_per_dataset[$dataset]}
        min_selected_features=${min_selected_features_per_dataset[$dataset]}
        max_selected_features=${max_selected_features_per_dataset[$dataset]}
        
        # dec_in 和 c_out 通常等于 enc_in
        dec_in=$enc_in
        c_out=$enc_in
        c_in=$enc_in

        # 根据数据集设置不同的预测长度
        if [ "$dataset" = "national_illness" ]; then
            seq_len=36
            pred_lens=(24 36 48 60)
            label_len=18
        elif [[ "$dataset" =~ ^PEMS ]]; then
            # PEMS数据集使用特殊的预测长度
            seq_len=$seq_len_meta
            pred_lens=(12 24 48 96)
            label_len=$label_len_meta
        else
            seq_len=$seq_len_meta
            pred_lens=("${pred_lens_meta[@]}")
            label_len=$label_len_meta
        fi

        log_dir="./logs/$model_name/$testname_with_vmd/$dataset"
        mkdir -p "$log_dir"

        for i in "${!pred_lens[@]}"; do
            pred_len=${pred_lens[$i]}
            train_epoch=${epochs[$i]}
            log_file="$log_dir/${model_name}_${dataset}_${seq_len}_${pred_len}_vmd${vmd_scales}_${current_time}.log"

            echo "Running: $model_name on $dataset | vmd_scales=$vmd_scales, lr=$learning_rate, bs=$batch_size, pred_len=$pred_len"
            python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path ./dataset/ \
                --data_path "$data_path" \
                --model_id "${dataset}_${seq_len}_${pred_len}_vmd${vmd_scales}" \
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
                --vmd_scales "$vmd_scales" \
                --vmd_alpha 1000 \
                --vmd_tau 0.2 \
                --vmd_dc 0 \
                --vmd_tol 1e-8 \
                --vmd_n_iter 500 \
                --vmd_init uniform \
                --vmd_chunk_size 8 \
                --feature_selection_threshold 0.001 \
                --lasso_max_iter 100 \
                --lasso_alpha 0.0001 \
                --lasso_lr 0.01 \
                --min_selected_features "$min_selected_features" \
                --max_selected_features "$max_selected_features" \
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
done

# === 汇总所有结果 ===
echo "=========================================================="
echo "All training tasks are completed. Generating results tables..."
echo "=========================================================="

# 为每个vmd_scales值生成单独的结果表
for vmd_scales in "${vmd_scales_list[@]}"; do
    testname_with_vmd="${testname}_vmd${vmd_scales}"
    log_base_dir="./logs/$model_name/$testname_with_vmd"
    output_csv_path="$log_base_dir/${model_name}_vmd${vmd_scales}_results.csv"
    
    if [ -d "$log_base_dir" ]; then
        echo "Generating results table for vmd_scales=$vmd_scales..."
        python generate_table.py --log_dir "$log_base_dir" --output_csv_path "$output_csv_path"
    fi
done

# 生成一个汇总所有vmd_scales结果的综合表格
echo "Generating combined results table for all vmd_scales values..."
combined_output_csv="./logs/$model_name/${testname}_all_vmd_scales_results.csv"

# 这里假设您有一个可以合并多个CSV文件的脚本
# 如果没有，您需要创建一个或者手动合并
# 这里提供一个简单的合并思路
first_vmd=true
for vmd_scales in "${vmd_scales_list[@]}"; do
    testname_with_vmd="${testname}_vmd${vmd_scales}"
    csv_file="./logs/$model_name/$testname_with_vmd/${model_name}_vmd${vmd_scales}_results.csv"
    
    if [ -f "$csv_file" ]; then
        if [ "$first_vmd" = true ]; then
            # 第一个文件，复制并添加vmd_scales列
            awk -v vs="$vmd_scales" 'BEGIN{FS=OFS=","} NR==1{print $0",vmd_scales"} NR>1{print $0","vs}' "$csv_file" > "$combined_output_csv"
            first_vmd=false
        else
            # 后续文件，追加并添加vmd_scales列
            awk -v vs="$vmd_scales" 'BEGIN{FS=OFS=","} NR>1{print $0","vs}' "$csv_file" >> "$combined_output_csv"
        fi
    fi
done

echo "All results tables have been generated."
echo "Individual tables: ./logs/$model_name/${testname}_vmd[1-5]/"
echo "Combined table: $combined_output_csv"