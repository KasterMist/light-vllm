import torch
from torch.utils.cpp_extension import load
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "Ada"  # 设置支持的CUDA架构列表
# 统一 PyTorch 扩展缓存目录，避免 sudo/非 sudo 使用不同 HOME 时反复编译或权限问题
# 使用仓库根目录下的 .torch_extensions 作为默认缓存目录
# 注意：如需自定义，请在进程启动前设置 TORCH_EXTENSIONS_DIR 环境变量覆盖
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../../"))
default_ext_dir = os.path.join(repo_root, ".torch_extensions")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", default_ext_dir)
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

# 获取当前文件所在的目录（已在上方定义）

cuda_kernels = load(
    name="lightvllm_gemm_kernels",
    sources=[os.path.join(current_dir, "gemm.cu")],
    verbose=True,
    extra_cuda_cflags=["-lineinfo"],  # Enable source correlation for Nsight Compute
)

def gemm_f32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    return cuda_kernels.sgemm_sliced_k_f32(a, b, c)
    