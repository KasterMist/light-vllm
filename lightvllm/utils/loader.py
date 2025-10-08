import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认的权重加载函数。
    
    当一个模型参数（nn.Parameter）没有指定特殊的加载方式时，会使用此函数。
    它的功能很简单：直接将从文件中加载的权重张量(`loaded_weight`)的数据
    完整地复制到模型参数(`param`)中。
    
    Args:
        param (nn.Parameter): 模型中需要被加载权重的参数。
        loaded_weight (torch.Tensor): 从权重文件中读取的张量。
    """
    assert param.size() == loaded_weight.size(), \
        f"权重尺寸不匹配: 模型参数尺寸 {param.size()}, 文件中权重尺寸 {loaded_weight.size()}"
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str):
    """
    从指定路径加载模型权重到给定的PyTorch模型中。
    
    这个函数的核心是处理张量并行（Tensor Parallelism）下的权重加载。
    在张量并行中，一些大的权重（如Q, K, V投影矩阵）会被切分并存储，
    加载时需要将这些切片（shards）正确地合并到模型的一个组合参数中。
    
    Args:
        model (nn.Module): 需要加载权重的PyTorch模型实例。
        path (str): 包含`.safetensors`权重文件的目录路径。
    """
    
    # packed_modules_mapping 是一个定义在模型中的字典，用于声明如何处理权重的切分和合并。
    # 它的格式通常是: { "原始权重名的一部分": ("模型中的目标参数名", 分片ID) }
    # 例如: {'q_proj': ('qkv_proj', 0), 'k_proj': ('qkv_proj', 1)}
    # 这表示文件中名为 '...q_proj.weight' 的权重，应被加载到模型参数 'qkv_proj' 的第 0 个分片。
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 使用glob找到路径下所有的.safetensors文件
    for file in glob(os.path.join(path, "*safetensors")):
        # 使用safetensors库安全地打开权重文件。
        # "pt"表示以PyTorch张量的格式读取，"cpu"表示先将权重加载到CPU内存中。
        with safe_open(file, "pt", "cpu") as f:
            # 遍历文件中的每一个权重张量名称
            for weight_name in f.keys():
                
                is_packed = False
                # 检查当前权重是否属于需要特殊处理的“打包”权重
                for packed_key in packed_modules_mapping:
                    if packed_key in weight_name:
                        # --- 处理打包/切分的权重 ---
                        is_packed = True
                        
                        # 从映射中获取目标参数名和分片ID
                        target_param_name, shard_id = packed_modules_mapping[packed_key]
                        
                        # 构建模型中实际的参数名称
                        # 例如, 'model.layers.0.self_attn.q_proj.weight' -> 'model.layers.0.self_attn.qkv_proj.weight'
                        param_name = weight_name.replace(packed_key, target_param_name)
                        
                        # 从模型中获取这个组合参数（例如 qkv_proj.weight）
                        param = model.get_parameter(param_name)
                        
                        # 获取附加到该参数上的自定义权重加载函数。
                        # 对于需要合并分片的参数，这个函数知道如何根据shard_id将权重放到正确的位置。
                        # 这个weight_loader是在模型定义时被附加到参数上的。
                        weight_loader = getattr(param, "weight_loader")
                        
                        # 执行加载函数。
                        # 这里是函数式编程思想的体现：weight_loader是一个变量，它指向一个函数。
                        # 这行代码执行该函数，传入目标参数、从文件中加载的权重分片和分片ID。
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        
                        break  # 找到匹配的映射后，处理完毕，跳出内层循环
                
                if not is_packed:
                    # --- 处理普通（未切分）的权重 ---
                    
                    # 直接根据权重名称从模型中获取对应的参数
                    param = model.get_parameter(weight_name)
                    
                    # 尝试获取该参数上附加的weight_loader。
                    # 如果没有（对于绝大多数普通参数），则使用我们定义的 default_weight_loader。
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    
                    # 执行加载函数。对于普通参数，这通常就是调用default_weight_loader，
                    # 它会简单地将整个权重张量复制到模型参数中。
                    weight_loader(param, f.get_tensor(weight_name))


# load_model_2 当前项目中并未使用。它的逻辑与 load_model 类似，但实现方式略有不同，
# 试图通过集合操作来提高效率。注释将同样予以保留以供参考。
def load_model_2(model: nn.Module, path: str):
    """
    (备用) 改进版的模型权重加载函数。
    
    相比原版load_model函数的改进：
    1. 逻辑更清晰：分两个阶段处理，先处理所有需要映射的权重，再处理普通权重。
    2. 效率可能更高：避免了对每个权重都遍历一次映射表。
    3. 避免重复处理：使用集合(set)来跟踪已处理的权重。
    
    Args:
        model: 要加载权重的PyTorch模型。
        path: 包含权重文件的目录路径。
    """
    
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {}) 
    
    for file in glob(os.path.join(path, "*safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            # 获取文件中所有可用的权重名称
            available_weights = set(f.keys())
            
            # === 第一阶段：处理需要映射的权重 ===
            processed_weights = set()
            for mapping_key, (target_param_name, shard_id) in packed_modules_mapping.items():
                # 找到所有包含当前映射键的权重
                matched_weights = [w for w in available_weights if mapping_key in w]
                
                for weight_name in matched_weights:
                    # 构建目标参数名
                    param_name = weight_name.replace(mapping_key, target_param_name)
                    param = model.get_parameter(param_name)
                    
                    # 获取并使用自定义加载器
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, f.get_tensor(weight_name), shard_id)
                    
                    # 将处理过的权重加入集合
                    processed_weights.add(weight_name)
            
            # === 第二阶段：处理剩余的未映射权重（使用标准流程） ===
            remaining_weights = available_weights - processed_weights
            for weight_name in remaining_weights:
                param = model.get_parameter(weight_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, f.get_tensor(weight_name))


