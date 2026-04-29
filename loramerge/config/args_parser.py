import os
import yaml
import logging
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    from peft.config import CONFIG_NAME
except ImportError:
    CONFIG_NAME = "adapter_config.json"  # 默认值


def load_yaml_config(config_path: str = "examples/merge_demo.yaml") -> dict:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return {}
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return {}


def validate_config(config: dict) -> bool:
    """校验配置合法性"""
    errors = []

    if not config.get("merge_only_lora", False):
        base_model_path = config.get("base_model_path", "")
        if not base_model_path:
            errors.append("base_model_path 不能为空（除非设置 merge_only_lora: true）")
        elif not os.path.exists(base_model_path):
            errors.append(f"基础模型路径不存在: {base_model_path}")

    lora_path_list = config.get("lora_path_list", [])
    if not lora_path_list:
        errors.append("lora_path_list 不能为空")
    else:
        merge_only_lora_flag = config.get("merge_only_lora", False)
        has_valid_lora_config = False
        for i, lora_path in enumerate(lora_path_list):
            if not lora_path:
                errors.append(f"第 {i+1} 个LoRA路径为空")
                continue
                
            if not os.path.exists(lora_path):
                errors.append(f"LoRA路径不存在: {lora_path}")
                continue
                
            # 检查是否有有效的 LoRA 配置文件
            if not os.path.isfile(lora_path):
                config_file = os.path.join(lora_path, CONFIG_NAME)
                if os.path.exists(config_file):
                    has_valid_lora_config = True
        
        # 仅合并LoRA模式下，至少需要一个完整的 LoRA 目录
        if merge_only_lora_flag and not has_valid_lora_config:
            errors.append(
                f"仅合并LoRA模式需要至少一个完整的 LoRA 目录\n"
                f"请确保至少有一个 LoRA 路径是完整目录（包含 {CONFIG_NAME}）\n"
                f"或者取消'merge_only_lora'选项"
            )

    if len(lora_path_list) == 1:
        if config.get("merge_only_lora", False):
            errors.append("单个LoRA不支持仅合并LoRA模式，需要提供基础模型")
    elif len(lora_path_list) >= 2:
        merge_method = config.get("merge_method", "")
        valid_methods = ["linear", "ties", "dare", "slerp"]
        if merge_method not in valid_methods:
            errors.append(f"merge_method 必须是以下之一: {valid_methods}")

    output_dir = config.get("output_dir", "")
    if not output_dir:
        errors.append("output_dir 不能为空")

    weights = config.get("weights", None)
    if weights is not None:
        if len(weights) != len(lora_path_list):
            errors.append("weights 数量必须与 lora_path_list 数量一致")
        elif any(w <= 0 for w in weights):
            errors.append("weights 必须都是正数")

    if config.get("merge_method") == "slerp" and len(lora_path_list) != 2:
        errors.append("SLERP算法仅支持2个LoRA的融合")

    alpha = config.get("alpha", 0.7)
    if alpha < 0 or alpha > 1:
        errors.append("alpha必须在0到1之间")

    dropout_rate = config.get("dropout_rate", 0.5)
    if dropout_rate < 0 or dropout_rate >= 1:
        errors.append("dropout_rate必须在0到1之间（不包括1）")

    slerp_t = config.get("slerp_t", 0.5)
    if slerp_t < 0 or slerp_t > 1:
        errors.append("slerp_t必须在0到1之间")

    if errors:
        logger.error("配置验证失败:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    logger.info("配置验证通过")
    return True


def check_lora_compatibility(config: dict) -> bool:
    """LoRA兼容性校验"""
    from ..core.lora_merger import load_lora_config
    
    lora_path_list = config.get("lora_path_list", [])
    if len(lora_path_list) < 2:
        return True
    
    logger.info("正在检查 LoRA 兼容性...")
    
    # 找到第一个有效的 LoRA 配置
    first_config = None
    first_valid_path = None
    for lora_path in lora_path_list:
        if not os.path.isfile(lora_path):
            config_file = os.path.join(lora_path, CONFIG_NAME)
            if os.path.exists(config_file):
                try:
                    first_config = load_lora_config(lora_path)
                    first_valid_path = lora_path
                    break
                except Exception as e:
                    logger.warning(f"尝试加载配置失败: {lora_path}, 错误: {e}")
                    continue
    
    if not first_config:
        logger.warning("没有找到有效的 LoRA 配置文件，跳过兼容性检查")
        return True
    
    compatible = True
    for i, lora_path in enumerate(lora_path_list, start=1):
        if lora_path == first_valid_path:
            continue
            
        # 只检查有配置文件的 LoRA
        if os.path.isfile(lora_path):
            logger.debug(f"LoRA {i} 是单文件，跳过兼容性检查: {lora_path}")
            continue
            
        config_file = os.path.join(lora_path, CONFIG_NAME)
        if not os.path.exists(config_file):
            logger.debug(f"LoRA {i} 没有配置文件，跳过兼容性检查: {lora_path}")
            continue
            
        try:
            current_config = load_lora_config(lora_path)
            
            if current_config.base_model_name_or_path != first_config.base_model_name_or_path:
                logger.warning(f"LoRA {i} 的基础模型与第一个不同")
                compatible = False
            
            if hasattr(current_config, 'r') and hasattr(first_config, 'r'):
                if current_config.r != first_config.r:
                    logger.warning(f"LoRA {i} 的 r 值与第一个不同")
                    compatible = False
        except Exception as e:
            logger.error(f"加载LoRA {i}配置失败: {str(e)}")
            compatible = False
    
    if compatible:
        logger.info("所有LoRA兼容性检查通过")
    
    return compatible