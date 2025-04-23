import pandas as pd
import json
from datetime import datetime

def calculate_deltas(df):
    """计算delta矩阵"""
    deltas = pd.DataFrame(0, index=df.index, columns=df.columns[1:])  # 第一行全0
    for i in range(1, len(df)):
        for col in df.columns[1:]:
            prev_mask = 1 if pd.notna(df.at[i-1, col]) else 0
            if prev_mask == 1:
                deltas.at[i, col] = 1  # 假设时间间隔为1小时
            else:
                deltas.at[i, col] = deltas.at[i-1, col] + 1
    return deltas

def process_data(input_file, output_file):
    df = pd.read_csv(input_file)
    samples = []
    deltas = calculate_deltas(df)  # 计算delta矩阵

    for i in range(0, len(df), 24):
        sample_df = df.iloc[i:i+24]
        delta_sample = deltas.iloc[i:i+24]
        forward = []

        for idx, (row, delta_row) in enumerate(zip(sample_df.iterrows(), delta_sample.iterrows())):
            _, data_row = row
            _, delta_data = delta_row
            evals = []
            masks = []
            values = []
            eval_masks = []
            forwards = []

            for col in data_row.index[1:]:  # 跳过datetime列
                val = data_row[col]
                if pd.notna(val):
                    masks.append(1)
                    values.append(float(val))
                    evals.append(float(val))
                    eval_masks.append(1)
                    forwards.append(float(val))
                else:
                    masks.append(0)
                    values.append(0.0)
                    evals.append(0.0)
                    eval_masks.append(0)
                    forwards.append(0.0)

            forward_entry = {
                'evals': evals,
                'deltas': list(delta_data),
                'forwards': forwards,
                'masks': masks,
                'values': values,
                'eval_masks': eval_masks
            }
            forward.append(forward_entry)

        sample_json = {
            'forward': forward,
            'label': 0,  # 可根据实际情况修改
            'is_train': 0.0  # 可根据实际情况修改
        }
        samples.append(sample_json)

    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_file = 'D:\Project\GAN for ts imputation\Generative-Semi-supervised-Learning-for-Multivariate-Time-Series-Imputation-main\datasets\AirQuality\pm25_ground.txt'  # 替换为实际输入文件名
    output_file = 'output.json'
    process_data(input_file, output_file)