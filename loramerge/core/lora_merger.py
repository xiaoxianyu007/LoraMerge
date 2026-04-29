import os
import signal
import sys
import logging
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model_state_dict, LoraConfig

def setup_logging(log_level: str = "INFO"):
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


_log_level = os.environ.get("LORAMERGE_LOG_LEVEL", "INFO")
logger = setup_logging(_log_level)


def set_log_level(log_level: str):
    global logger
    logger = setup_logging(log_level)
    logger.info(f"日志级别已设置为: {log_level.upper()}")

fusion_interrupted = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用计算设备: {device}")
_model_cache = {}
_signal_handler_registered = False


def signal_handler(sig, frame):
    global fusion_interrupted
    fusion_interrupted = True
    logger.info("\n[中断] LoRA 融合已停止！")
    sys.exit(0)


def register_signal_handler():
    global _signal_handler_registered
    if _signal_handler_registered:
        return
    try:
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal_handler)
            _signal_handler_registered = True
            logger.debug("信号处理器已注册")
    except Exception as e:
        logger.debug(f"无法注册信号处理器（可能在子线程中）: {e}")


register_signal_handler()


def load_base_model(base_model_path: str, torch_dtype: torch.dtype = torch.float16):
    logger.info(f"加载基础模型: {base_model_path}")
    try:
        # 确保本地路径存在（完全离线模式）
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(
                f"本地模型路径不存在: {base_model_path}\n"
                f"请确保基础模型已正确下载到本地\n"
                f"当前工作目录: {os.getcwd()}"
            )
        
        # Monkey patch huggingface_hub 的验证函数
        from huggingface_hub.utils import _validators
        original_validate_repo_id = _validators.validate_repo_id
        
        def patched_validate_repo_id(repo_id, *args, **kwargs):
            logger.debug(f"离线模式：跳过验证: {repo_id}")
            return
        
        _validators.validate_repo_id = patched_validate_repo_id
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, 
                local_files_only=True
            )
        finally:
            _validators.validate_repo_id = original_validate_repo_id
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("基础模型加载成功")
        return model, tokenizer
    except Exception as e:
        logger.error(f"加载基础模型失败: {str(e)}")
        raise


def load_lora_config(lora_path: str) -> PeftConfig:
    from peft.config import CONFIG_NAME
    
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA 路径不存在: {lora_path}")
    
    if os.path.isfile(lora_path):
        raise ValueError(
            f"提供的路径 '{lora_path}' 是一个 safetensor 文件\n"
            f"使用单文件时，需要提供 base_model_path\n"
            f"系统将自动从基础模型推断配置。"
        )
    
    config_file = os.path.join(lora_path, CONFIG_NAME)
    if not os.path.exists(config_file):
        raise ValueError(
            f"在 LoRA 目录 '{lora_path}' 中找不到 {CONFIG_NAME}\n"
            f"请确保这是一个有效的 LoRA 目录。"
        )
    
    try:
        return PeftConfig.from_pretrained(lora_path)
    except Exception as e:
        logger.error(f"加载LoRA配置失败: {str(e)}")
        raise


def load_lora_weights(lora_path: str, base_model_path: str = None, base_model=None) -> Dict[str, torch.Tensor]:
    from peft.config import CONFIG_NAME
    
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA 路径不存在: {lora_path}")
    
    # 如果是单文件 safetensor
    if os.path.isfile(lora_path):
        return _load_lora_from_single_file(lora_path, base_model_path, base_model)
    
    config_file = os.path.join(lora_path, CONFIG_NAME)
    if not os.path.exists(config_file):
        raise ValueError(
            f"在 LoRA 目录 '{lora_path}' 中找不到 {CONFIG_NAME}\n"
            f"请确保这是一个有效的 LoRA 目录，包含完整的 LoRA 文件。"
        )
    
    config = PeftConfig.from_pretrained(lora_path)
    
    # 如果没有提供基础模型路径且没有提供基础模型对象，直接加载权重
    if base_model_path is None and base_model is None:
        logger.info(f"仅合并LoRA模式：直接加载权重: {lora_path}")
        # 使用 safetensors 直接加载权重文件
        import glob
        safetensors_files = glob.glob(os.path.join(lora_path, "*.safetensors"))
        bin_files = glob.glob(os.path.join(lora_path, "adapter_model.bin"))
        
        state_dict = {}
        if safetensors_files:
            from safetensors.torch import load_file
            for safetensor_file in safetensors_files:
                state_dict.update(load_file(safetensor_file))
        elif bin_files:
            for bin_file in bin_files:
                state_dict.update(torch.load(bin_file, map_location='cpu', weights_only=True))
        else:
            raise FileNotFoundError(f"在 {lora_path} 中找不到权重文件")
        
        logger.info(f"LoRA权重加载完成，共 {len(state_dict)} 个参数")
        return state_dict
    
    base_model_name = base_model_path if base_model_path else config.base_model_name_or_path
    
    logger.info(f"加载LoRA权重: {lora_path} (基础模型: {base_model_name})")
    
    try:
        if base_model is not None:
            logger.info("使用已提供的基础模型")
            model_to_use = base_model
        elif base_model_name in _model_cache:
            logger.info(f"从缓存加载基础模型: {base_model_name}")
            model_to_use = _model_cache[base_model_name]
        else:
            logger.info(f"首次加载基础模型: {base_model_name}")
            
            # 判断是否是本地路径
            is_local_path = (
                base_model_name.startswith('/') or 
                base_model_name.startswith('./') or 
                base_model_name.startswith('../') or
                (os.sep in base_model_name and not base_model_name.count('/') <= 2)
            )
            
            logger.info(f"使用本地路径加载模型: {base_model_name}")
            
            if not os.path.exists(base_model_name):
                raise FileNotFoundError(
                    f"本地模型路径不存在: {base_model_name}\n"
                    f"请确保基础模型已正确下载到本地\n"
                    f"当前工作目录: {os.getcwd()}"
                )
            
            from huggingface_hub.utils import _validators
            original_validate_repo_id = _validators.validate_repo_id
            
            def patched_validate_repo_id(repo_id, *args, **kwargs):
                logger.debug(f"离线模式：跳过验证: {repo_id}")
                return
            
            _validators.validate_repo_id = patched_validate_repo_id
            
            try:
                model_to_use = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )
            finally:
                _validators.validate_repo_id = original_validate_repo_id
            
            _model_cache[base_model_name] = model_to_use
        
        peft_model = PeftModel.from_pretrained(model_to_use, lora_path)
        state_dict = get_peft_model_state_dict(peft_model)
        
        if base_model is None:
            del peft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"LoRA权重加载完成，共 {len(state_dict)} 个参数")
        return state_dict
    except Exception as e:
        logger.error(f"加载LoRA权重失败: {str(e)}")
        raise


def _load_lora_from_single_file(safetensor_path: str, base_model_path: str = None, base_model=None) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    
    logger.info(f"检测到单个 safetensor 文件: {safetensor_path}")
    
    try:
        state_dict = load_file(safetensor_path, device="cpu")
        logger.info(f"从 safetensor 文件加载了 {len(state_dict)} 个参数")
        
        has_lora_format = any(".lora_" in key or ".lora_" in key.lower() for key in state_dict.keys())
        
        if not has_lora_format:
            logger.warning("权重文件中未检测到 LoRA 格式，可能需要额外的转换步骤")
        
        return state_dict
    except Exception as e:
        logger.error(f"从 safetensor 文件加载权重失败: {str(e)}")
        raise


def clear_model_cache():
    global _model_cache
    for model in _model_cache.values():
        del model
    _model_cache = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("模型缓存已清空")


def merge_linear(lora_weights_list: List[Dict[str, torch.Tensor]], weights: Optional[List[float]] = None, **kwargs) -> Dict[str, torch.Tensor]:
    logger.info("使用线性融合算法")
    
    if weights is None:
        weights = [1.0 / len(lora_weights_list)] * len(lora_weights_list)
    
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("权重总和必须大于0")
    
    normalized_weights = [w / total_weight for w in weights]
    
    merged = {}
    for key in lora_weights_list[0].keys():
        merged[key] = sum(
            w * lora_weights[key].to(torch.float32).to(device)
            for w, lora_weights in zip(normalized_weights, lora_weights_list)
        )
    
    return merged


def merge_ties(lora_weights_list: List[Dict[str, torch.Tensor]], alpha: float = 0.7, **kwargs) -> Dict[str, torch.Tensor]:
    logger.info(f"使用 TIES-MERGE 算法 (alpha={alpha})")
    
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha必须在0到1之间")
    
    merged = {}
    for key in lora_weights_list[0].keys():
        tensors = [lw[key].to(torch.float32).to(device) for lw in lora_weights_list]
        
        signs = [torch.sign(t) for t in tensors]
        consensus_sign = torch.prod(torch.stack(signs), dim=0)
        
        weighted_sum = sum(t * consensus_sign for t in tensors) / len(tensors)
        
        threshold = alpha * torch.mean(torch.abs(torch.stack(tensors)))
        merged[key] = weighted_sum * (torch.abs(weighted_sum) > threshold)
    
    return merged


def merge_dare(lora_weights_list: List[Dict[str, torch.Tensor]], dropout_rate: float = 0.5, **kwargs) -> Dict[str, torch.Tensor]:
    logger.info(f"使用 DARE-MERGE 算法 (dropout_rate={dropout_rate})")
    
    if dropout_rate < 0 or dropout_rate >= 1:
        raise ValueError("dropout_rate必须在0到1之间（不包括1）")
    
    merged = {}
    for key in lora_weights_list[0].keys():
        tensors = [lw[key].to(torch.float32).to(device) for lw in lora_weights_list]
        
        mask = torch.bernoulli(torch.ones_like(tensors[0]) * (1 - dropout_rate))
        
        selected = [t * mask for t in tensors]
        merged[key] = sum(selected) / max(torch.sum(mask), 1e-6)
    
    return merged


def merge_slerp(lora_weights_list: List[Dict[str, torch.Tensor]], t: float = 0.5, **kwargs) -> Dict[str, torch.Tensor]:
    logger.info(f"使用 SLERP 算法 (t={t})")
    
    if t < 0 or t > 1:
        raise ValueError("t必须在0到1之间")
    
    if len(lora_weights_list) != 2:
        logger.warning("SLERP算法仅支持2个LoRA，自动退化为线性融合")
        return merge_linear(lora_weights_list)
    
    merged = {}
    for key in lora_weights_list[0].keys():
        t1, t2 = lora_weights_list[0][key].to(torch.float32).to(device), lora_weights_list[1][key].to(torch.float32).to(device)
        
        norm1 = torch.norm(t1)
        norm2 = torch.norm(t2)
        
        t1_normalized = t1 / norm1 if norm1 > 1e-6 else t1
        t2_normalized = t2 / norm2 if norm2 > 1e-6 else t2
        
        dot = torch.sum(t1_normalized * t2_normalized)
        dot = torch.clamp(dot, -1, 1)
        theta = torch.arccos(dot)
        
        sin_theta = torch.sin(theta)
        if sin_theta < 1e-6:
            merged[key] = (1 - t) * t1 + t * t2
        else:
            merged[key] = (
                torch.sin((1 - t) * theta) / sin_theta * t1_normalized +
                torch.sin(t * theta) / sin_theta * t2_normalized
            ) * ((1 - t) * norm1 + t * norm2)
    
    return merged


def merge_only_lora(lora_weights_list: List[Dict[str, torch.Tensor]], merge_method: str = "linear", 
                   weights: Optional[List[float]] = None, **kwargs) -> Dict[str, torch.Tensor]:
    logger.info(f"仅合并 LoRA 模块，方法: {merge_method}")
    
    merge_functions = {
        "linear": merge_linear,
        "ties": merge_ties,
        "dare": merge_dare,
        "slerp": merge_slerp
    }
    
    if merge_method not in merge_functions:
        raise ValueError(f"未知的融合方法: {merge_method}")
    
    return merge_functions[merge_method](lora_weights_list, weights=weights, **kwargs)


def merge_lora_to_base(base_model, lora_weights: Dict[str, torch.Tensor], lora_config: PeftConfig) -> nn.Module:
    logger.info("将 LoRA 权重合并到基础模型...")
    
    try:
        peft_model = PeftModel(base_model, lora_config)
        peft_model.load_state_dict(lora_weights, strict=False)
        merged_model = peft_model.merge_and_unload()
        
        logger.info("LoRA权重合并到基础模型成功")
        return merged_model
    except Exception as e:
        logger.error(f"合并LoRA到基础模型失败: {str(e)}")
        raise


def merge_single_lora_to_base(base_model_path: str, lora_path: str, output_dir: str) -> str:
    logger.info(f"开始将单个LoRA合并到基础模型")
    logger.info(f"  - 基础模型: {base_model_path}")
    logger.info(f"  - LoRA: {lora_path}")
    logger.info(f"  - 输出目录: {output_dir}")
    
    model, tokenizer = load_base_model(base_model_path)
    
    peft_model = PeftModel.from_pretrained(model, lora_path)
    merged_model = peft_model.merge_and_unload()
    
    save_merged_model(output_dir, merged_model, tokenizer)
    
    return output_dir


def save_merged_lora(output_dir: str, merged_weights: Dict[str, torch.Tensor], lora_config: PeftConfig):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
    timestamp_dir = os.path.join(output_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    
    from safetensors.torch import save_file
    save_file(merged_weights, os.path.join(timestamp_dir, "adapter_model.safetensors"))
    
    lora_config.save_pretrained(timestamp_dir)
    
    logger.info(f"合并后的 LoRA 已保存到: {timestamp_dir}")
    return timestamp_dir


def save_merged_model(output_dir: str, model, tokenizer):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
    timestamp_dir = os.path.join(output_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    
    model.save_pretrained(timestamp_dir)
    tokenizer.save_pretrained(timestamp_dir)
    
    logger.info(f"合并后的模型已保存到: {timestamp_dir}")


def run_merge_from_yaml():
    from ..config.args_parser import load_yaml_config, validate_config
    
    if len(sys.argv) < 3:
        logger.error("用法: loramerge-cli merge <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[2]
    config = load_yaml_config(config_path)
    
    if not validate_config(config):
        logger.error("配置文件验证失败！")
        sys.exit(1)
    
    start_merge(config)


def _try_load_lora_config(lora_path: str):
    """尝试加载 LoRA 配置，如果失败返回 None"""
    try:
        return load_lora_config(lora_path)
    except Exception as e:
        logger.debug(f"无法加载 LoRA 配置: {lora_path}, 错误: {e}")
        return None


def start_merge(config: dict) -> str:
    global fusion_interrupted
    fusion_interrupted = False
    
    base_model_path = config.get("base_model_path", "")
    lora_path_list = config.get("lora_path_list", [])
    merge_method = config.get("merge_method", "linear")
    output_dir = config.get("output_dir", "./merged_lora")
    merge_only_lora_flag = config.get("merge_only_lora", False)
    weights = config.get("weights", None)
    
    alpha = config.get("alpha", 0.7)
    dropout_rate = config.get("dropout_rate", 0.5)
    slerp_t = config.get("slerp_t", 0.5)
    
    logger.info(f"开始 LoRA 融合")
    logger.info(f"  - 基础模型: {'(仅合并LoRA)' if merge_only_lora_flag else base_model_path}")
    logger.info(f"  - LoRA数量: {len(lora_path_list)}")
    logger.info(f"  - 融合方法: {merge_method}")
    logger.info(f"  - 输出目录: {output_dir}")
    
    if len(lora_path_list) == 1:
        if merge_only_lora_flag:
            error_msg = "单个LoRA不支持仅合并LoRA模式，需要提供基础模型"
            logger.error(error_msg)
            return f"错误：{error_msg}"
        if not base_model_path:
            error_msg = "单个LoRA合并需要提供基础模型路径"
            logger.error(error_msg)
            return f"错误：{error_msg}"
        
        logger.info("单个LoRA合并到基础模型...")
        return merge_single_lora_to_base(base_model_path, lora_path_list[0], output_dir)
    
    from peft.config import CONFIG_NAME
    
    first_lora_config = None
    valid_lora_config_path = None
    for lora_path in lora_path_list:
        if not os.path.isfile(lora_path):
            config_file = os.path.join(lora_path, CONFIG_NAME)
            if os.path.exists(config_file):
                valid_lora_config_path = lora_path
                break
    
    # 如果找到有效配置，进行兼容性检查
    if valid_lora_config_path and not merge_only_lora_flag:
        from ..config.args_parser import check_lora_compatibility
        check_lora_compatibility(config)
    
    logger.info("加载 LoRA 权重...")
    lora_weights_list = []
    for i, lora_path in enumerate(lora_path_list):
        logger.info(f"  [{i+1}/{len(lora_path_list)}] {lora_path}")
        lora_weights_list.append(load_lora_weights(lora_path, base_model_path if not merge_only_lora_flag else None))
        
        if fusion_interrupted:
            return "融合已中断"
    
    # 尝试从有效路径加载配置
    if valid_lora_config_path:
        first_lora_config = _try_load_lora_config(valid_lora_config_path)
    
    logger.info("执行融合...")
    merged_weights = merge_only_lora(
        lora_weights_list, 
        merge_method=merge_method, 
        weights=weights,
        alpha=alpha,
        dropout_rate=dropout_rate,
        t=slerp_t
    )
    
    if fusion_interrupted:
        return "融合已中断"
    
    if merge_only_lora_flag:
        if first_lora_config is None:
            error_msg = "使用单文件 LoRA 时不支持仅合并 LoRA 模式，需要提供基础模型"
            logger.error(error_msg)
            return f"错误：{error_msg}"
        save_merged_lora(output_dir, merged_weights, first_lora_config)
    else:
        model, tokenizer = load_base_model(base_model_path)
        if first_lora_config is not None:
            merged_model = merge_lora_to_base(model, merged_weights, first_lora_config)
        else:
            merged_model = merge_lora_weights_to_model(model, merged_weights)
        save_merged_model(output_dir, merged_model, tokenizer)
    
    return output_dir


def merge_lora_weights_to_model(base_model, lora_weights: Dict[str, torch.Tensor]) -> nn.Module:
    """直接将权重合并到模型（用于单文件 LoRA）"""
    logger.info("直接合并权重到基础模型...")
    try:
        missing_keys, unexpected_keys = base_model.load_state_dict(lora_weights, strict=False)
        if missing_keys:
            logger.warning(f"缺少的权重键: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"意外的权重键: {unexpected_keys}")
        return base_model
    except Exception as e:
        logger.error(f"合并权重到基础模型失败: {str(e)}")
        raise


def stop_merge():
    global fusion_interrupted
    fusion_interrupted = True
    logger.info("准备停止 LoRA 融合...")