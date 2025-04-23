import json
import numpy as np
import os
from pathlib import Path
from scipy.optimize import minimize

class NumpyEncoder(json.JSONEncoder):
    """处理 NumPy 数据类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def optimize_latent_space(train_value, label_value, max_iter=100):
    """
    使用LDO方法优化空气质量数据
    
    Args:
        train_value: 原始训练值
        label_value: 标签值
        max_iter: 最大迭代次数
    
    Returns:
        优化后的值
    """
    def objective(z):
        # 目标函数：最小化与标签值的差异
        # 确保使用标量值进行计算
        z_scalar = z[0] if isinstance(z, np.ndarray) else z
        label_scalar = label_value[0] if isinstance(label_value, np.ndarray) else label_value
        return (z_scalar - label_scalar) ** 2
    
    # 使用训练值作为初始值
    initial_value = train_value[0] if isinstance(train_value, np.ndarray) else train_value
    z0 = np.array([initial_value])
    
    # 优化
    result = minimize(objective, z0, method='Nelder-Mead', 
                     options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
    
    # 确保结果在合理范围内
    optimized_value = int(round(result.x[0]))  # 转换为整数
    # 使用加权平均来平滑结果
    alpha = 0.7  # 权重因子
    train_scalar = train_value[0] if isinstance(train_value, np.ndarray) else train_value
    smoothed_value = int(round(alpha * optimized_value + (1 - alpha) * train_scalar))  # 转换为整数
    
    return smoothed_value

def reconstruct_air_quality_data(train_data, label_data):
    """
    重构空气质量数据，只处理第一个数据（空气质量数据）
    
    Args:
        train_data: 原始训练数据数组
        label_data: 标签数据数组
    
    Returns:
        重构后的训练数据数组
    """
    try:
        # 将数据转换为numpy数组
        train_array = np.array(train_data)
        label_array = np.array(label_data)
        
        # 检查数组形状
        print(f"Train array shape: {train_array.shape}")
        print(f"Label array shape: {label_array.shape}")
        
        # 初始化重构数据
        reconstructed_data = []
        
        # 对每个时间步进行处理
        for i in range(len(train_array)):
            time_step_data = []
            for j in range(len(train_array[i])):
                try:
                    train_point = train_array[i][j]
                    label_point = label_array[i][j]
                    
                    # 获取原始三元组数据
                    train_value = train_point[0]
                    train_mask = train_point[1]
                    train_delay = train_point[2]
                    
                    label_value = label_point[0]
                    
                    # 使用LDO方法优化空气质量数据
                    optimized_value = optimize_latent_space(train_value, label_value)
                    
                    # 创建新的数据点，保持原有的掩码值和时滞值
                    time_step_data.append([
                        int(optimized_value),
                        int(train_point[1]),  # 保持原始掩码值
                        int(train_point[2])   # 保持原始时滞值
                    ])
                    
                except Exception as e:
                    print(f"Error processing point at time {i}, position {j}: {str(e)}")
                    # 发生错误时使用原始值
                    time_step_data.append([
                        int(train_point[0]),
                        int(train_point[1]),
                        int(train_point[2])
                    ])
                    continue
            
            reconstructed_data.append(time_step_data)
        
        return reconstructed_data
    except Exception as e:
        print(f"Error in reconstruct_air_quality_data: {str(e)}")
        print(f"Train data sample: {train_data[:3] if train_data else 'No data'}")
        print(f"Label data sample: {label_data[:3] if label_data else 'No data'}")
        raise

def load_and_process_pair(train_file, label_file):
    """
    加载并处理一对train和label文件
    
    Args:
        train_file: train数据文件路径
        label_file: label数据文件路径
    
    Returns:
        处理后的数据字典
    """
    try:
        # 加载数据
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(label_file, 'r') as f:
            label_data = json.load(f)
        
        # 检查数据格式
        print(f"\nProcessing file: {train_file}")
        
        # 重构训练数据
        reconstructed_array = reconstruct_air_quality_data(train_data, label_data)
        
        # 统计信息
        train_array = np.array(train_data)
        label_array = np.array(label_data)
        reconstructed_np = np.array(reconstructed_array)
        
        # 创建处理数据结构
        processed_data = {
            "reconstructed_train_data": reconstructed_array,
            "metadata": {
                "data_length": int(len(train_data)),
                "time_steps": int(len(train_data[0])),
                "optimization_stats": {
                    "original_mean": int(round(np.mean(train_array[:, :, 0]))),
                    "original_std": int(round(np.std(train_array[:, :, 0]))),
                    "target_mean": int(round(np.mean(label_array[:, :, 0]))),
                    "target_std": int(round(np.std(label_array[:, :, 0]))),
                    "optimized_mean": int(round(np.mean(reconstructed_np[:, :, 0]))),
                    "optimized_std": int(round(np.std(reconstructed_np[:, :, 0])))
                }
            }
        }
        
        return processed_data
    except Exception as e:
        print(f"Error in load_and_process_pair: {str(e)}")
        print(f"Train data sample: {train_data[:3] if train_data else 'No data'}")
        print(f"Label data sample: {label_data[:3] if label_data else 'No data'}")
        raise

def process_dataset(train_dir, label_dir, output_dir):
    """
    处理整个数据集
    
    Args:
        train_dir: train数据目录
        label_dir: label数据目录
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有train文件
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.json')])
    
    for train_file in train_files:
        try:
            # 构建对应的label文件路径
            label_file = train_file
            train_path = os.path.join(train_dir, train_file)
            label_path = os.path.join(label_dir, label_file)
            
            # 检查文件是否存在
            if not os.path.exists(label_path):
                print(f"Warning: Label file {label_file} not found, skipping {train_file}")
                continue
            
            # 处理文件对
            processed_data = load_and_process_pair(train_path, label_path)
            
            # 只保存重构的数据数组
            output_file = os.path.join(output_dir, f"optimized_{train_file}")
            with open(output_file, 'w') as f:
                json.dump(processed_data["reconstructed_train_data"], f, separators=(',', ':'), cls=NumpyEncoder)
            
            print(f"Successfully processed {train_file}")
            
        except Exception as e:
            continue

if __name__ == "__main__":
    # 设置目录路径
    train_dir = "./train"
    label_dir = "./label"
    output_dir = "./processed"
    
    # 处理数据集
    process_dataset(train_dir, label_dir, output_dir) 