import os
from glob import glob  # 用于文件路径模式匹配，查找符合条件的文件
import torch
from torch import nn
from safetensors import safe_open  # 安全地打开和读取safetensors格式的权重文件

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认的权重加载函数
    直接将加载的权重复制到参数中
    """
    param.data.copy_(loaded_weight)  # 将加载的权重数据复制到模型参数中

def load_model(model: nn.Module, path: str):
    """
    从指定路径加载模型权重
    
    Args:
        model: 要加载权重的PyTorch模型
        path: 包含权重文件的目录路径
    """
    
    # 获取模型的打包模块映射表（如果存在）
    # packed_modules_mapping用于处理权重名称的映射和分片
    # 格式: {原始名称: (新名称, 分片ID)}
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {}) 
    # 遍历指定路径下所有的safetensors文件
    for file in glob(os.path.join(path, "*safetensors")):
        # 安全地打开safetensors文件，指定设备为CPU
        with safe_open(file, "pt", "cpu") as f:
            # 遍历文件中的每个权重张量名称
            for weight_name in f.keys():
                
                # 检查当前权重名称是否需要进行packed module映射
                for k in packed_modules_mapping:
                    if k in weight_name:  # 如果权重名称包含需要映射的键
                        # 获取映射信息：新的参数名称和分片ID
                        v, shard_id = packed_modules_mapping[k]
                        
                        # 将权重名称中的旧名称替换为新名称
                        param_name = weight_name.replace(k, v)
                        
                        # 根据新的参数名称获取模型中对应的参数
                        param = model.get_parameter(param_name)
                        
                        # 获取参数的自定义权重加载器
                        weight_loader = getattr(param, "weight_loader")
                        
                        # 使用自定义加载器加载权重，传入分片ID
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        
                        break  # 找到映射后跳出循环
                else:
                    # 如果没有找到packed module映射，使用标准流程
                    
                    # 直接根据权重名称获取模型参数
                    param = model.get_parameter(weight_name)
                    
                    # 获取参数的权重加载器，如果没有则使用默认加载器
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    
                    # 加载权重（不需要分片ID）
                    weight_loader(param, f.get_tensor(weight_name))


def load_model_2(model: nn.Module, path: str):
    """
    改进版的模型权重加载函数
    
    相比原版load_model函数的改进：
    1. 逻辑更清晰：分两个阶段处理映射权重和标准权重
    2. 效率更高：避免对每个权重都遍历映射表
    3. 避免重复处理：使用集合跟踪已处理的权重
    
    Args:
        model: 要加载权重的PyTorch模型
        path: 包含权重文件的目录路径
    """
    
    # 获取模型的打包模块映射表（如果存在）
    # packed_modules_mapping用于处理权重名称的映射和分片
    # 格式: {原始名称: (新名称, 分片ID)}
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {}) 
    
    # 遍历指定路径下所有的safetensors文件
    for file in glob(os.path.join(path, "*safetensors")):
        # 安全地打开safetensors文件，指定设备为CPU
        with safe_open(file, "pt", "cpu") as f:
            # 获取文件中所有权重名称
            available_weights = set(f.keys())
            
            # === 第一阶段：处理需要映射的权重 ===
            processed_weights = set()
            for mapping_key, (target_param_name, shard_id) in packed_modules_mapping.items():
                # 查找包含映射键的权重名称
                matched_weights = [w for w in available_weights if mapping_key in w]
                
                for weight_name in matched_weights:
                    # 将权重名称中的旧名称替换为新名称
                    param_name = weight_name.replace(mapping_key, target_param_name)
                    
                    # 根据新的参数名称获取模型中对应的参数
                    param = model.get_parameter(param_name)
                    
                    # 获取参数的自定义权重加载器
                    weight_loader = getattr(param, "weight_loader")
                    
                    # 使用自定义加载器加载权重，传入分片ID
                    weight_loader(param, f.get_tensor(weight_name), shard_id)
                    
                    # 标记为已处理
                    processed_weights.add(weight_name)
            
            # === 第二阶段：处理剩余的未映射权重（使用标准流程） ===
            remaining_weights = available_weights - processed_weights
            for weight_name in remaining_weights:
                # 直接根据权重名称获取模型参数
                param = model.get_parameter(weight_name)
                
                # 获取参数的权重加载器，如果没有则使用默认加载器
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                
                # 加载权重（不需要分片ID）
                weight_loader(param, f.get_tensor(weight_name))


