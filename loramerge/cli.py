# Copyright 2025 LoraMerge Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import NoReturn

USAGE = (
    "-" * 70
    + "\n"
    + "| LoraMerge - Advanced LoRA Fusion Tool                         |\n"
    + "| Usage:                                                         |\n"
    + "|   loramerge-cli webui: launch Web UI for LoRA fusion           |\n"
    + "|   loramerge-cli merge: merge LoRAs via YAML config             |\n"
    + "|   loramerge-cli help: show help information                    |\n"
    + "-" * 70
)


def main() -> NoReturn:
    """
    顶层命令调度入口
    仿照 LLaMA Factory 架构设计
    """
    # 在主线程中注册信号处理器
    from .core.lora_merger import register_signal_handler
    register_signal_handler()

    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(0)

    command = sys.argv.pop(1)

    # 启动 WebUI 界面
    if command == "webui":
        from .webui.gradio_ui import run_webui
        run_webui()

    # 命令行模式融合（读取YAML）
    elif command == "merge":
        from .core.lora_merger import run_merge_from_yaml
        run_merge_from_yaml()

    # 帮助信息
    elif command == "help":
        print(USAGE)

    # 未知命令
    else:
        print(f"Unknown command: {command}\n{USAGE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
