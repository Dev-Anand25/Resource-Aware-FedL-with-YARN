import subprocess
from pathlib import Path
from typing import List

def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")

def hdfs_mkdir(hdfs_path: str) -> None:
    run(["hdfs", "dfs", "-mkdir", "-p", hdfs_path])

def hdfs_put(local_path: str, hdfs_path: str) -> None:
    run(["hdfs", "dfs", "-put", "-f", local_path, hdfs_path])

def hdfs_get(hdfs_path: str, local_path: str) -> None:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    run(["hdfs", "dfs", "-get", "-f", hdfs_path, local_path])

def hdfs_exists(hdfs_path: str) -> bool:
    p = subprocess.run(["hdfs", "dfs", "-test", "-e", hdfs_path])
    return p.returncode == 0

def hdfs_ls(hdfs_path: str) -> str:
    p = subprocess.run(["hdfs", "dfs", "-ls", hdfs_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return ""
    return p.stdout
