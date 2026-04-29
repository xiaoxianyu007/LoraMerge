from setuptools import setup, find_packages

setup(
    name="loramerge",
    version="0.1.0",
    author="LoraMerge Team",
    description="Advanced LoRA Fusion Tool with WebUI",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "peft>=0.11.0",
        "gradio>=4.0",
        "pyyaml",
        "accelerate",
        "safetensors",
    ],
    entry_points={
        "console_scripts": [
            "loramerge-cli = loramerge.cli:main",
        ]
    },
)
