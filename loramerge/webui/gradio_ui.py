import os
import sys
import socket
import logging
import gradio as gr
from typing import List, Dict, Optional

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


def parse_lora_paths(lora_path_list: List[str]) -> List[str]:
    return [p.strip() for p in lora_path_list if p.strip()]


def parse_weights(weights_str: str) -> Optional[List[float]]:
    if not weights_str or weights_str.strip() == "":
        return None
    try:
        weights = [float(w.strip()) for w in weights_str.split(",")]
        return weights
    except Exception as e:
        logger.error(f"解析权重错误: {e}")
        return None


def merge_loras(
    base_model: str,
    merge_only: bool,
    lora_count: int,
    lora1: str,
    lora2: str,
    lora3: str,
    lora4: str,
    lora5: str,
    method: str,
    w1: float,
    w2: float,
    w3: float,
    w4: float,
    w5: float,
    alpha: float,
    dropout_rate: float,
    slerp_t: float,
    output: str
) -> str:
    logger.info(f"接收到融合请求: method={method}, lora_count={lora_count}")

    try:
        if lora_count == 1:
            if not base_model:
                return "❌ 错误：单个LoRA合并需要提供基础模型路径"
            if not lora1.strip():
                return "❌ 错误：请输入LoRA路径"

            # 验证单个LoRA路径
            if not os.path.exists(lora1):
                return f"❌ 错误：LoRA路径不存在: {lora1}"

            logger.info(f"单个LoRA合并到基础模型: {lora1}")
            from ..core.lora_merger import merge_single_lora_to_base
            result = merge_single_lora_to_base(base_model, lora1, output)
            return f"✅ 单个LoRA合并完成！\n输出目录: {result}"

        lora_paths = [lora1, lora2, lora3, lora4, lora5][:lora_count]
        lora_path_list = parse_lora_paths(lora_paths)
        logger.info(f"解析到 {len(lora_path_list)} 个LoRA路径")

        if len(lora_path_list) < 2:
            return "❌ 错误：至少需要2个LoRA文件"

        if merge_only:
            # 仅合并LoRA模式下，至少需要一个完整的 LoRA 目录
            has_valid_lora_config = False
            for i, lora_path in enumerate(lora_path_list):
                if not os.path.isfile(lora_path):
                    # 检查是否有配置文件
                    config_file = os.path.join(lora_path, CONFIG_NAME)
                    if os.path.exists(config_file):
                        has_valid_lora_config = True
                        break
            
            if not has_valid_lora_config:
                return (
                    f"❌ 错误：仅合并LoRA模式需要至少一个完整的 LoRA 目录\n"
                    f"请确保至少有一个 LoRA 路径是完整目录（包含 {CONFIG_NAME}）\n"
                    f"或者取消'仅合并LoRA'选项"
                )
        
        # 验证所有LoRA路径
        for i, lora_path in enumerate(lora_path_list):
            if not os.path.exists(lora_path):
                return f"❌ 错误：第 {i+1} 个LoRA路径不存在: {lora_path}"

        weights_list = [w1, w2, w3, w4, w5][:lora_count]

        config = {
            "base_model_path": base_model if not merge_only else "",
            "lora_path_list": lora_path_list,
            "merge_method": method,
            "output_dir": output,
            "merge_only_lora": merge_only,
            "weights": weights_list,
            "alpha": alpha,
            "dropout_rate": dropout_rate,
            "slerp_t": slerp_t
        }

        logger.info("正在调用核心融合函数...")
        from ..core.lora_merger import start_merge
        result = start_merge(config)

        if result == "融合已中断":
            return "⏹️ 融合已中断"
        elif result.startswith("错误"):
            return f"❌ {result}"
        else:
            return f"✅ 融合完成！\n输出目录: {result}"
            
    except Exception as e:
        logger.error(f"融合失败: {e}", exc_info=True)
        return f"❌ 融合失败: {str(e)}"


def stop_merge_fn() -> str:
    try:
        from ..core.lora_merger import stop_merge
        stop_merge()
        return "⏹️ 正在停止融合..."
    except Exception as e:
        logger.error(f"停止融合失败: {e}")
        return f"❌ 停止失败: {str(e)}"


def is_port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result = s.connect_ex(('127.0.0.1', port))
            return result != 0
    except Exception as e:
        logger.error(f"检查端口错误: {e}")
        return False


def run_webui():
    logger.info("正在启动 LoRA 融合界面...")
    
    port = 7860
    if not is_port_available(port):
        logger.warning(f"端口 {port} 已被占用，尝试其他端口...")
        for p in range(7861, 7871):
            if is_port_available(p):
                port = p
                logger.info(f"使用端口 {port}")
                break
    
    try:
        with gr.Blocks(title="LoRA Merge Tool") as demo:
            gr.Markdown("# 🎯 LoRA Merge Tool")
            gr.Markdown("高级 LoRA 融合工具 - 支持多种融合算法")
            gr.Markdown("""**📋 LoRA 使用说明**：基础模型需完整文件夹；LoRA 需包含 `adapter_model.safetensors` 和 `adapter_config.json`；配置内基座路径需与实际路径一致。""")
            
            gr.Markdown("### **基础模型设置**")
            base_model_path = gr.Textbox(
                label="基础模型路径",
                placeholder="输入基础模型目录路径..."
            )
            merge_only_lora = gr.Checkbox(
                label="**仅合并LoRA**（不合并到基础模型）",
                value=False
            )
            
            gr.Markdown("### **LoRA 设置**")
            lora_count = gr.Radio(
                label="LoRA 数量（双击选择）",
                choices=[1, 2, 3, 4, 5],
                value=2,
                interactive=True
            )
            
            lora1 = gr.Textbox(label="LoRA 1 路径", placeholder="/path/to/lora1")
            lora2 = gr.Textbox(label="LoRA 2 路径", placeholder="/path/to/lora2")
            lora3 = gr.Textbox(label="LoRA 3 路径", placeholder="/path/to/lora3", visible=False)
            lora4 = gr.Textbox(label="LoRA 4 路径", placeholder="/path/to/lora4", visible=False)
            lora5 = gr.Textbox(label="LoRA 5 路径", placeholder="/path/to/lora5", visible=False)
            
            gr.Markdown("### **融合算法(双击选择)**")
            
            with gr.Row():
                method_linear = gr.Button("线性融合", variant="secondary", size="sm")
                method_ties = gr.Button("TIES-MERGE", variant="secondary", size="sm")
                method_dare = gr.Button("DARE-MERGE", variant="secondary", size="sm")
                method_slerp = gr.Button("SLERP", variant="secondary", size="sm")
            
            current_method = gr.State("")
            
            with gr.Row():
                linear_params = gr.Column(visible=False)
                with linear_params:
                    gr.Markdown("#### 线性融合参数")
                    w1 = gr.Number(label="LoRA 1 权重", value=0.5, precision=2)
                    w2 = gr.Number(label="LoRA 2 权重", value=0.5, precision=2)
                    w3 = gr.Number(label="LoRA 3 权重", value=0.33, precision=2, visible=False)
                    w4 = gr.Number(label="LoRA 4 权重", value=0.25, precision=2, visible=False)
                    w5 = gr.Number(label="LoRA 5 权重", value=0.20, precision=2, visible=False)
                
                ties_params = gr.Column(visible=False)
                with ties_params:
                    gr.Markdown("#### TIES-MERGE 参数")
                    ties_alpha = gr.Slider(
                        label="alpha",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.7,
                        info="权重阈值，低于此值的权重将被置零"
                    )
                
                dare_params = gr.Column(visible=False)
                with dare_params:
                    gr.Markdown("#### DARE-MERGE 参数")
                    dare_dropout = gr.Slider(
                        label="dropout_rate",
                        minimum=0.0,
                        maximum=0.9,
                        step=0.05,
                        value=0.5,
                        info="随机丢弃率，控制权重选择的随机性"
                    )
                
                slerp_params = gr.Column(visible=False)
                with slerp_params:
                    gr.Markdown("#### SLERP 参数")
                    slerp_t = gr.Slider(
                        label="t",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                        info="插值系数，0表示完全使用第一个LoRA，1表示完全使用第二个"
                    )
            
            gr.Markdown("### **输出设置**")
            output_dir = gr.Textbox(
                label="输出目录",
                value="./merged_lora"
            )
            
            with gr.Row():
                merge_btn = gr.Button("开始融合", variant="primary", size="lg")
                stop_btn = gr.Button("停止融合", variant="stop", size="lg")
            
            status_output = gr.Textbox(
                label="融合状态",
                lines=8,
                interactive=False
            )
            
            def select_method(method: str, count: int):
                uniform_weight = round(1.0 / count, 2)
                
                linear_visible = (method == "linear")
                ties_visible = (method == "ties")
                dare_visible = (method == "dare")
                slerp_visible = (method == "slerp")
                
                linear_style = "primary" if method == "linear" else "secondary"
                ties_style = "primary" if method == "ties" else "secondary"
                dare_style = "primary" if method == "dare" else "secondary"
                slerp_style = "primary" if method == "slerp" else "secondary"
                
                slerp_locked = (method == "slerp")
                
                if slerp_locked:
                    lora_count_value = 2
                    lora_count_interactive = False
                    lora3_visible = False
                    lora4_visible = False
                    lora5_visible = False
                    w3_visible = False
                    w4_visible = False
                    w5_visible = False
                else:
                    lora_count_value = count
                    lora_count_interactive = True
                    lora3_visible = (count >= 3)
                    lora4_visible = (count >= 4)
                    lora5_visible = (count >= 5)
                    w3_visible = (linear_visible and count >= 3)
                    w4_visible = (linear_visible and count >= 4)
                    w5_visible = (linear_visible and count >= 5)
                
                return (
                    gr.update(visible=linear_visible),
                    gr.update(visible=ties_visible),
                    gr.update(visible=dare_visible),
                    gr.update(visible=slerp_visible),
                    gr.update(variant=linear_style),
                    gr.update(variant=ties_style),
                    gr.update(variant=dare_style),
                    gr.update(variant=slerp_style),
                    method,
                    gr.update(value=lora_count_value, interactive=lora_count_interactive),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=lora3_visible),
                    gr.update(visible=lora4_visible),
                    gr.update(visible=lora5_visible),
                    gr.update(visible=linear_visible, value=uniform_weight),
                    gr.update(visible=linear_visible, value=uniform_weight),
                    gr.update(visible=w3_visible, value=uniform_weight),
                    gr.update(visible=w4_visible, value=uniform_weight),
                    gr.update(visible=w5_visible, value=uniform_weight)
                )
            
            def toggle_base_model(merge_only: bool):
                return gr.update(
                    interactive=not merge_only,
                    placeholder="" if merge_only else "输入基础模型目录路径..."
                )
            
            merge_only_lora.change(
                fn=toggle_base_model,
                inputs=[merge_only_lora],
                outputs=[base_model_path]
            )
            
            def handle_lora_count_change(count: int):
                uniform_weight = round(1.0 / count, 2)
                
                if count == 1:
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(interactive=False, variant="secondary"),
                        gr.update(interactive=False, variant="secondary"),
                        gr.update(interactive=False, variant="secondary"),
                        gr.update(interactive=False, variant="secondary"),
                        "single_lora",
                        gr.update(interactive=True),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
                else:
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(value=None),
                        gr.update(interactive=True),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=True if count >= 3 else False),
                        gr.update(visible=True if count >= 4 else False),
                        gr.update(visible=True if count >= 5 else False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
            
            lora_count.change(
                fn=handle_lora_count_change,
                inputs=[lora_count],
                outputs=[linear_params, ties_params, dare_params, slerp_params,
                        method_linear, method_ties, method_dare, method_slerp, current_method, lora_count,
                        lora1, lora2, lora3, lora4, lora5, w1, w2, w3, w4, w5]
            )
            
            method_linear.click(
                fn=lambda c: select_method("linear", c),
                inputs=[lora_count],
                outputs=[linear_params, ties_params, dare_params, slerp_params,
                        method_linear, method_ties, method_dare, method_slerp, current_method, lora_count,
                        lora1, lora2, lora3, lora4, lora5, w1, w2, w3, w4, w5]
            )
            method_ties.click(
                fn=lambda c: select_method("ties", c),
                inputs=[lora_count],
                outputs=[linear_params, ties_params, dare_params, slerp_params,
                        method_linear, method_ties, method_dare, method_slerp, current_method, lora_count,
                        lora1, lora2, lora3, lora4, lora5, w1, w2, w3, w4, w5]
            )
            method_dare.click(
                fn=lambda c: select_method("dare", c),
                inputs=[lora_count],
                outputs=[linear_params, ties_params, dare_params, slerp_params,
                        method_linear, method_ties, method_dare, method_slerp, current_method, lora_count,
                        lora1, lora2, lora3, lora4, lora5, w1, w2, w3, w4, w5]
            )
            method_slerp.click(
                fn=lambda c: select_method("slerp", c),
                inputs=[lora_count],
                outputs=[linear_params, ties_params, dare_params, slerp_params,
                        method_linear, method_ties, method_dare, method_slerp, current_method, lora_count,
                        lora1, lora2, lora3, lora4, lora5, w1, w2, w3, w4, w5]
            )
            
            merge_btn.click(
                fn=merge_loras,
                inputs=[
                    base_model_path,
                    merge_only_lora,
                    lora_count,
                    lora1, lora2, lora3, lora4, lora5,
                    current_method,
                    w1, w2, w3, w4, w5,
                    ties_alpha,
                    dare_dropout,
                    slerp_t,
                    output_dir
                ],
                outputs=[status_output]
            )
            
            stop_btn.click(
                fn=stop_merge_fn,
                outputs=[status_output]
            )
        
        logger.info(f"准备启动 Gradio 服务，端口: {port}...")
        debug_mode = os.environ.get("LORAMERGE_DEBUG", "false").lower() == "true"
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            debug=debug_mode
        )
        logger.info("Gradio 服务已启动")
        
    except Exception as e:
        logger.error(f"启动失败: {e}", exc_info=True)