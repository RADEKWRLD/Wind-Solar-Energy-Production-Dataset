"""
统一入口：按顺序运行全部任务脚本

使用方式：python3 main.py
"""
import subprocess
import sys
import os
from datetime import datetime

SCRIPTS = [
    "01_eda.py",
    "02_regression.py",
    "03_classification.py",
    "04_clustering.py",
    "05_time_series.py",
]

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for script in SCRIPTS:
        name = script.replace(".py", "")
        log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
        print(f"\n运行 {script}，日志保存到 {log_path}")

        with open(log_path, "w") as log:
            result = subprocess.run(
                [sys.executable, os.path.join(root, "scripts", script)],
                cwd=root, stdout=log, stderr=subprocess.STDOUT
            )
            if result.returncode != 0:
                print(f"  {script} 失败（退出码 {result.returncode}）")
