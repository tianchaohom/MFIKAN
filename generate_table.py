import os
import re
import pandas as pd
import argparse

def parse_log_files(log_dir):
    results = []
    for dataset_name in os.listdir(log_dir):
        dataset_path = os.path.join(log_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        for file in os.listdir(dataset_path):
            if not file.endswith('.log'):
                continue

            log_file = os.path.join(dataset_path, file)
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    last_line = lines[-1].strip() if lines else ""

                mse_match = re.search(r"mse:\s*([0-9.eE+-]+)", last_line)
                mae_match = re.search(r"mae:\s*([0-9.eE+-]+)", last_line)

                if not mse_match or not mae_match:
                    print(f"⚠️ Could not find the MSE or MAE values in {file}")
                    continue

                mse = round(float(mse_match.group(1)), 3)
                mae = round(float(mae_match.group(1)), 3)

                # 解析文件名
                stem = os.path.splitext(file)[0]
                parts = stem.split('_')
                if len(parts) < 4:
                    print(f"⚠️ File name format mismatch: {file}")
                    continue

                model_name = parts[0]
                pred_len = int(parts[-3])  

                results.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Pred_len': pred_len,
                    'MSE': mse,
                    'MAE': mae
                })

            except Exception as e:
                print(f"❌ Error processing file {file}: {e}")

    return results

def create_and_display_table_from_logs(log_dir, output_csv_path=None):
  
    raw_data = parse_log_files(log_dir)

    if not raw_data:
        print("No valid log data found. Please check if the log_dir is correct.")
        return

    df = pd.DataFrame(raw_data)
    unique_models = df['Model'].unique()
    if len(unique_models) > 1:
        print(f"⚠️ Warning: Multiple models found {unique_models}, but this script only supports one model. The table will display data for the first model only.")
        model_name = unique_models[0]
        df = df[df['Model'] == model_name]
    elif len(unique_models) == 1:
        model_name = unique_models[0]
    else:
        print("No model names found.")
        return

    df = df.sort_values(by=['Dataset', 'Pred_len'])

    all_rows = []
    
    for dataset, group in df.groupby('Dataset'):
        for _, row in group.iterrows():
            all_rows.append({
                'Dataset': dataset,
                'Metric': row['Pred_len'],
                'MSE': row['MSE'],
                'MAE': row['MAE']
            })
        
        avg_mse = round(group['MSE'].mean(), 3)
        avg_mae = round(group['MAE'].mean(), 3)
        all_rows.append({
            'Dataset': dataset,
            'Metric': 'Avg',
            'MSE': avg_mse,
            'MAE': avg_mae
        })

    final_df = pd.DataFrame(all_rows)

    final_df = final_df.set_index(['Dataset', 'Metric'])
    final_df.columns = pd.MultiIndex.from_product([[model_name], ['MSE', 'MAE']])
    pd.options.display.float_format = '{:.3f}'.format

    print(final_df.to_string())

    if output_csv_path:
        df_to_save = final_df.copy()
        df_to_save.reset_index(inplace=True)
        df_to_save.columns = ['Models', 'Metric', 'MSE', 'MAE']
        df_to_save.to_csv(output_csv_path, index=False, float_format='%.3f')
        print(f"\nThe table has been successfully saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a formatted table from log files.")
    parser.add_argument('--log_dir', type=str, required=True, help="The root directory of the log files.")
    parser.add_argument('--output_csv_path', type=str, default=None, help="Optional path to save the CSV file.")
    
    args = parser.parse_args()
    
    create_and_display_table_from_logs(args.log_dir, args.output_csv_path)