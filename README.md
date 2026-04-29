# 🎯 LoRA Merge Tool

LoRA 融合工具，支持多种融合算法，提供 WebUI 界面和命令行两种操作方式。

## ✨ 功能特性

- 🚀 **多种融合算法**：支持 Linear、TIES-MERGE、DARE-MERGE、SLERP 四种算法
- 🖥️ **WebUI 界面**：基于 Gradio 的可视化操作界面，支持动态参数显示
- 📝 **命令行模式**：支持 YAML 配置文件批量处理
- 🎯 **灵活合并**：支持仅合并 LoRA 模块或合并到基础模型
- 🔢 **多LoRA支持**：支持1-5个LoRA的融合
- 📦 **单文件支持**：支持单独的 `.safetensors` 文件作为输入
- ⚡ **中断支持**：支持随时中断融合过程
- 🎛️ **参数自适应**：根据LoRA数量自动调整权重初始化
- 💾 **时间戳输出**：每次融合自动创建时间戳文件夹，避免覆盖
- 📱 **离线模式**：完全离线运行，不依赖外部网络
- 🔧 **GPU加速**：自动检测并使用GPU加速融合计算

## 🔧 安装方法

```bash
# 克隆仓库（或下载源码）
git clone <repository-url>
cd LoraMerge

# 安装依赖（开发模式）
pip install -e .
```

### 依赖要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| torch | >= 2.0 | PyTorch 深度学习框架 |
| transformers | >= 4.40 | Hugging Face Transformers |
| peft | >= 0.11.0 | Parameter-Efficient Fine-Tuning |
| gradio | >= 4.0 | WebUI 界面框架 |
| pyyaml | - | YAML 配置文件解析 |
| accelerate | - | 分布式训练支持 |
| safetensors | - | 安全的权重文件格式 |

## 🚀 使用方式

### 1. WebUI 模式

```bash
loramerge-cli webui
```

启动后访问 `http://localhost:7860` 即可使用可视化界面。

### 2. 命令行模式

```bash
loramerge-cli merge examples/merge_demo.yaml
```

### 3. 帮助信息

```bash
loramerge-cli help
```

## 📋 配置文件说明

创建 YAML 配置文件：

```yaml
# ==============================================
# 基础模型配置
# ==============================================
# 基础模型路径（当 merge_only_lora 为 false 时必须填写）
base_model_path: "/path/to/base/model"

# ==============================================
# 融合模式配置
# ==============================================
# 是否仅合并LoRA模块（不合并到基础模型）
# true: 只输出合并后的LoRA文件
# false: 将融合后的LoRA合并到基础模型
merge_only_lora: false

# ==============================================
# LoRA 配置
# ==============================================
# LoRA路径列表
# - 支持完整目录（包含 adapter_config.json 和权重文件）
# - 支持单个 .safetensors 文件
# - 1个LoRA：直接合并到基础模型（merge_only_lora 必须为 false）
# - 2-5个LoRA：先融合再合并（或仅合并LoRA）
lora_path_list:
  - "/path/to/lora1"           # 完整目录
  - "/path/to/lora2.safetensors" # 单文件

# ==============================================
# 融合算法配置
# ==============================================
# 融合算法选择（多个LoRA时使用）
# 可选值: linear, ties, dare, slerp
merge_method: "linear"

# 融合权重（可选，不填则使用均匀权重）
# 权重数量必须与LoRA数量一致
weights:
  - 0.5
  - 0.5

# ==============================================
# 算法特定参数（可选）
# ==============================================
# alpha: 0.7           # TIES-MERGE 参数，权重阈值
# dropout_rate: 0.5    # DARE-MERGE 参数，随机丢弃率
# slerp_t: 0.5         # SLERP 参数，插值系数

# ==============================================
# 输出配置
# ==============================================
# 融合输出目录（会自动创建时间戳子目录）
output_dir: "./merged_lora"
```

## 🔬 融合算法说明

### Linear（线性融合）⭐⭐⭐⭐⭐

简单的加权平均方法，适合大多数场景，计算速度快。

**公式：**
```
merged = Σ(w_i * lora_i) / Σ(w_i)
```

**适用场景：**
- 需要快速融合多个LoRA
- LoRA之间特征互补
- 不确定使用哪种算法时的默认选择

**参数：**
- `weights`: 各LoRA的权重（可选，默认均匀分配）

---

### TIES-MERGE ⭐⭐⭐⭐

基于权重符号一致性的融合方法，保留符号一致的权重。

**核心思想：**
1. 计算所有LoRA权重的符号
2. 确定符号一致性（所有符号相同）
3. 仅保留符号一致的权重
4. 应用阈值过滤小权重

**特点：**
- 保留共识特征，去除冲突特征
- 适合需要保留特定风格的场景
- 自动过滤噪声权重

**参数：**
- `alpha`: 权重阈值（0-1，默认0.7），低于此值的权重将被置零

---

### DARE-MERGE ⭐⭐⭐⭐

基于 Dropout 的融合方法，随机选择权重进行合并。

**核心思想：**
1. 使用伯努利分布生成随机掩码
2. 根据掩码选择存活的权重
3. 对选中的权重进行平均合并

**特点：**
- 引入随机性，减少过拟合风险
- 适合多个LoRA的融合
- 增加模型鲁棒性

**参数：**
- `dropout_rate`: 随机丢弃率（0-1，默认0.5）

---

### SLERP（球面线性插值）⭐⭐⭐⭐⭐

球面线性插值，专为两个LoRA设计的平滑过渡方法。

**核心思想：**
1. 将权重向量归一化到单位球面
2. 在球面上进行线性插值
3. 保持向量范数不变

**特点：**
- 在单位球面上进行插值
- 保持向量范数不变
- 适合风格迁移和特征平滑过渡

**限制：**
- **仅支持两个LoRA**，多于两个时自动退化为线性融合

**参数：**
- `t`: 插值系数（0-1，默认0.5）
  - `t=0`: 完全使用第一个LoRA
  - `t=0.5`: 平均混合
  - `t=1`: 完全使用第二个LoRA

## 📁 项目结构

```
LoraMerge/
├── loramerge/                    # 主模块
│   ├── __init__.py               # 版本信息和模块导出
│   ├── cli.py                    # 命令行入口
│   ├── config/                   # 配置模块
│   │   ├── __init__.py
│   │   └── args_parser.py        # YAML配置解析和验证
│   ├── core/                     # 核心模块
│   │   ├── __init__.py
│   │   └── lora_merger.py        # 融合核心逻辑
│   └── webui/                    # WebUI模块
│       ├── __init__.py
│       └── gradio_ui.py          # Gradio界面实现
├── examples/                     # 示例配置
│   └── merge_demo.yaml           # 融合示例配置
├── README.md                     # 项目说明文档
├── requirements.txt              # 依赖列表
├── setup.py                      # 安装配置
└── pyproject.toml                # 项目元数据
```

## 🛠️ 命令行参数

```bash
loramerge-cli <command> [options]

Commands:
  webui    启动 Web UI 界面
  merge    使用 YAML 配置文件进行融合
  help     显示帮助信息

Options:
  --help   显示命令帮助
```

## 📊 使用流程

### WebUI 模式

#### 1. 启动界面
```bash
loramerge-cli webui
```

#### 2. 设置基础模型
- **仅合并LoRA模式**：勾选「仅合并LoRA（不合并到基础模型）」
- **合并到基础模型**：输入基础模型路径

#### 3. 设置LoRA数量
- 选择LoRA数量（1-5个）
- 系统自动显示对应数量的输入框

#### 4. 添加LoRA路径
- 输入各LoRA的路径
- 支持完整目录或单个 `.safetensors` 文件

#### 5. 选择融合算法
| 场景 | 算法选择 | 说明 |
|------|---------|------|
| 单个LoRA | 自动禁用 | 直接合并到基础模型 |
| 两个LoRA | Linear/TIES/DARE/SLERP | SLERP仅支持2个LoRA |
| 三个及以上 | Linear/TIES/DARE | SLERP自动退化 |

#### 6. 设置参数
- **Linear**: 设置各LoRA权重（默认均匀分配）
- **TIES-MERGE**: 设置alpha阈值（0-1）
- **DARE-MERGE**: 设置dropout_rate（0-1）
- **SLERP**: 设置插值系数t（0-1）

#### 7. 设置输出目录
- 指定合并结果保存位置
- 每次融合自动创建时间戳子目录

#### 8. 开始融合
- 点击「开始融合」按钮
- 可随时点击「停止融合」中断

### 命令行模式

#### 1. 创建配置文件
```bash
cp examples/merge_demo.yaml my_config.yaml
```

#### 2. 编辑配置文件
```bash
vim my_config.yaml
```

#### 3. 运行融合
```bash
loramerge-cli merge my_config.yaml
```

## 📁 输入输出格式

### 输入格式

#### LoRA 目录结构
```
lora_directory/
├── adapter_config.json    # 必须：LoRA配置文件
└── adapter_model.safetensors  # 必须：权重文件（或 .bin）
```

#### 单文件模式
```
/path/to/lora.safetensors
```

### 输出格式

#### 仅合并LoRA模式
```
output_dir/
└── 2026_0429_1457/           # 时间戳子目录
    ├── adapter_config.json    # 合并后的配置
    └── adapter_model.safetensors  # 合并后的权重
```

#### 合并到基础模型模式
```
output_dir/
└── 2026_0429_1457/           # 时间戳子目录
    ├── config.json            # 模型配置
    ├── pytorch_model.bin      # 合并后的模型权重
    ├── tokenizer.json         # 分词器配置
    └── ...                    # 其他相关文件
```

## 📝 使用示例

### 示例1：单个LoRA合并到基础模型

**配置文件：**
```yaml
base_model_path: "./models/Qwen/Qwen2.5-0.5B-Instruct"
merge_only_lora: false
lora_path_list:
  - "./lora/character_lora"
output_dir: "./merged_model"
```

**命令：**
```bash
loramerge-cli merge examples/single_lora.yaml
```

### 示例2：仅合并两个LoRA

**配置文件：**
```yaml
merge_only_lora: true
lora_path_list:
  - "./lora/chinese_style"
  - "./lora/anime_style.safetensors"
merge_method: "linear"
output_dir: "./merged_lora"
```

**命令：**
```bash
loramerge-cli merge examples/merge_loras.yaml
```

### 示例3：使用SLERP融合两个角色LoRA

**配置文件：**
```yaml
base_model_path: "./models/base_model"
merge_only_lora: false
lora_path_list:
  - "./lora/character_a"
  - "./lora/character_b"
merge_method: "slerp"
slerp_t: 0.5
output_dir: "./merged_model"
```

**命令：**
```bash
loramerge-cli merge examples/slerp_merge.yaml
```

### 示例4：三个LoRA使用TIES-MERGE融合

**配置文件：**
```yaml
base_model_path: "./models/base_model"
merge_only_lora: false
lora_path_list:
  - "./lora/character"
  - "./lora/scene"
  - "./lora/style"
merge_method: "ties"
alpha: 0.7
weights:
  - 0.4
  - 0.3
  - 0.3
output_dir: "./merged_model"
```

**命令：**
```bash
loramerge-cli merge examples/ties_merge.yaml
```

## 🛠️ 常见问题

### Q1：LoRA路径应该怎么填？

**A：**
- **完整目录**：包含 `adapter_config.json` 和权重文件的目录
- **单文件**：直接指向 `.safetensors` 文件
- **路径格式**：支持绝对路径和相对路径

### Q2：什么是"仅合并LoRA"模式？

**A：**
- 不加载基础模型，直接融合LoRA权重
- 输出合并后的LoRA文件（`adapter_config.json` + `adapter_model.safetensors`）
- 至少需要一个完整的LoRA目录（提供配置信息）

### Q3：为什么需要基础模型路径？

**A：**
- 单文件LoRA没有配置信息，需要从基础模型获取
- 合并到基础模型时需要加载基础模型
- 仅合并LoRA模式且所有LoRA都是完整目录时不需要

### Q4：SLERP为什么只能用于两个LoRA？

**A：**
SLERP（球面线性插值）是为两个向量设计的插值算法，在高维空间中难以扩展到多个向量。当LoRA数量超过2时，系统会自动退化为线性融合。

### Q5：输出文件为什么在时间戳目录里？

**A：**
为了避免覆盖之前的融合结果，每次融合都会在输出目录下创建一个以当前时间命名的子目录（格式：`YYYY_MMDD_HHMM`）。

### Q6：支持离线使用吗？

**A：**
是的！所有模型和LoRA都从本地路径加载，不依赖外部网络。

## 📊 性能对比

| 算法 | 时间复杂度 | 内存占用 | 适用场景 |
|------|-----------|---------|---------|
| Linear | O(n) | 低 | 快速融合 |
| TIES-MERGE | O(n) | 中 | 特征选择 |
| DARE-MERGE | O(n) | 中 | 鲁棒融合 |
| SLERP | O(1) | 低 | 两个LoRA过渡 |

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境

```bash
# 克隆仓库
git clone <repository-url>
cd LoraMerge

# 创建虚拟环境
conda create -n loramerge python=3.10
conda activate loramerge

# 安装依赖
pip install -e .[dev]
```

### 代码规范

- 使用 `black` 进行代码格式化
- 使用 `flake8` 进行代码检查
- 提交前运行测试

## 📄 许可证

Apache License 2.0

## 🙏 致谢

感谢以下开源项目：

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [Gradio](https://github.com/gradio-app/gradio)
- [Safetensors](https://github.com/huggingface/safetensors)
- [Llama Factory](https://github.com/hiyouga/LLaMA-Factory)

---

**注意**：使用前请确保已安装所有依赖，并具备足够的显存来加载模型。
