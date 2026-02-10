# server.py
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import flwr as fl
import numpy as np

from yarn_monitor import YarnMonitor
from hdfs_utils import hdfs_mkdir, hdfs_put, hdfs_get, hdfs_exists

import time
import csv
from pathlib import Path

# ---- CSV logging schema (stable!) ----
CSV_FIELDS = [
    "round", "phase", "t",

    # YARN (cluster-level)
    "total_mb", "allocated_mb", "free_mb",
    "total_vcores", "allocated_vcores", "free_vcores",

    # Spark sizing
    "spark_executors", "spark_executor_cores", "spark_executor_mem_gb",

    # Summary
    "agg_seconds", "num_clients_used",
]

def append_csv(path: str, row: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(path).exists()

    safe = {k: row.get(k, "") for k in CSV_FIELDS}  # fill missing keys

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(safe)

def params_to_ndarrays(parameters: fl.common.Parameters) -> List[np.ndarray]:
    return fl.common.parameters_to_ndarrays(parameters)

def ndarrays_to_params(ndarrays: List[np.ndarray]) -> fl.common.Parameters:
    return fl.common.ndarrays_to_parameters(ndarrays)

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

class ResourceAwareSparkFedAvg(fl.server.strategy.Strategy):
    def __init__(
        self,
        rm_url: str,
        hdfs_base: str = "hdfs:///fedl",
        min_clients: int = 2,
        fraction_fit: float = 1.0,
        cores_per_exec: int = 2,
        max_exec: int = 20,
        min_exec: int = 2,
        exec_mem_gb_cap: int = 8,
        spark_submit: str = "spark-submit",
        spark_agg_script: str = "spark_agg.py",
    ):
        self.rm = YarnMonitor(rm_url)
        self.hdfs_base = hdfs_base.rstrip("/")
        self.min_clients = min_clients
        self.fraction_fit = fraction_fit
        self.cores_per_exec = cores_per_exec
        self.max_exec = max_exec
        self.min_exec = min_exec
        self.exec_mem_gb_cap = exec_mem_gb_cap
        self.spark_submit = spark_submit
        self.spark_agg_script = spark_agg_script

        self.current_round = 0
        self.initial_parameters: Optional[fl.common.Parameters] = None

    # ---- Strategy API ----
    def initialize_parameters(self, client_manager):
        # Let clients initialize if not set
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        self.current_round = server_round
        # sample clients
        num_available = client_manager.num_available()
        num_sample = max(self.min_clients, int(self.fraction_fit * num_available))
        clients = client_manager.sample(num_clients=num_sample, min_num_clients=self.min_clients)
        fit_ins = fl.common.FitIns(parameters, config={"round": server_round})
        return [(c, fit_ins) for c in clients]

    def aggregate_fit(
    self,
    server_round: int,
    results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    failures,
    ):
        if len(results) < self.min_clients:
            return None, {}

    # -----------------------------
    # Prepare HDFS directories
    # -----------------------------
        round_dir = f"{self.hdfs_base}/rounds/r_{server_round}"
        upd_dir = f"{round_dir}/updates"
        meta_dir = f"{round_dir}/meta"
        out_dir = f"{self.hdfs_base}/global"
        hdfs_mkdir(upd_dir)
        hdfs_mkdir(meta_dir)
        hdfs_mkdir(out_dir)

        Path("./logs").mkdir(parents=True, exist_ok=True)
        log_path = "./logs/yarn_resource_log.csv"

        manifest = []

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # -----------------------------
            # Save client updates -> HDFS
            # -----------------------------
            for client, fitres in results:
                cid = client.cid
                nds = params_to_ndarrays(fitres.parameters)

                local_npz = td_path / f"client_{cid}.npz"
                # Make sure it stays numeric (no pickle)
                np.savez(local_npz, *nds)

                hdfs_npz_path = f"{upd_dir}/client_{cid}.npz"
                hdfs_put(str(local_npz), hdfs_npz_path)

                manifest.append(
                    {
                        "cid": cid,
                        "path": hdfs_npz_path,
                        "num_examples": int(fitres.num_examples),
                    }
                )

            # manifest.json -> HDFS
            local_manifest = td_path / "manifest.json"
            local_manifest.write_text(json.dumps(manifest, indent=2))
            hdfs_manifest_path = f"{meta_dir}/manifest.json"
            hdfs_put(str(local_manifest), hdfs_manifest_path)

            # -----------------------------
            # Resource-aware Spark sizing
            # -----------------------------
            free0 = self.rm.free_resources()
            free_vcores = int(free0["free_vcores"])
            free_mb = int(free0["free_mb"])

            est_exec = free_vcores // max(1, self.cores_per_exec)
            num_exec = clamp(est_exec, self.min_exec, self.max_exec)

            usable_mb = int(free_mb * 0.8)  # keep headroom
            mem_per_exec_mb = max(1024, usable_mb // max(1, num_exec))
            mem_per_exec_gb = clamp(mem_per_exec_mb // 1024, 1, self.exec_mem_gb_cap)

            if free_mb < 2048:
                num_exec = 1
                mem_per_exec_gb = 1

            hdfs_out = f"{out_dir}/global_round_{server_round}.npz"

            # IMPORTANT:
            # In YARN *cluster* mode, your executors run in YARN containers.
            # This python path must exist inside those containers (i.e., on the node).
            VENV_PY = "/home/anand/Projects/FedL/.venv/bin/python"

            cmd = [
                self.spark_submit,
                "--master", "yarn",
                "--deploy-mode", "cluster",
                "--name", f"fedl-agg-r{server_round}",

                "--conf", f"spark.executor.instances={num_exec}",
                "--conf", f"spark.executor.cores={self.cores_per_exec}",
                "--conf", f"spark.executor.memory={mem_per_exec_gb}g",

                # Force python for driver/executors (works only if path exists on nodes)
                "--conf", f"spark.pyspark.python={VENV_PY}",
                "--conf", f"spark.pyspark.driver.python={VENV_PY}",
                "--conf", f"spark.executorEnv.PYSPARK_PYTHON={VENV_PY}",
                "--conf", f"spark.yarn.appMasterEnv.PYSPARK_PYTHON={VENV_PY}",

                self.spark_agg_script,
                "--round", str(server_round),
                "--manifest", hdfs_manifest_path,
                "--out", hdfs_out,
            ]

            print("\n[Server] YARN free (pre-size):", free0)
            print("[Server] Launching:", " ".join(cmd), "\n")

            # -----------------------------
            # BEFORE Spark job
            # -----------------------------
            free_before = self.rm.free_resources()

            append_csv(log_path, {
                "round": server_round,
                "phase": "before",
                "t": 0.0,

                # YARN memory
                "total_mb": free_before["total_mb"],
                "allocated_mb": free_before["allocated_mb"],
                "free_mb": free_before["free_mb"],

                # YARN vcores
                "total_vcores": free_before["total_vcores"],
                "allocated_vcores": free_before["allocated_vcores"],
                "free_vcores": free_before["free_vcores"],

                # Spark config (planned)
                "spark_executors": num_exec,
                "spark_executor_cores": self.cores_per_exec,
                "spark_executor_mem_gb": mem_per_exec_gb,
            })

            # -----------------------------
            # DURING Spark job (poll)
            # -----------------------------
            t0 = time.time()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            while p.poll() is None:
                s = self.rm.free_resources()

                append_csv(log_path, {
                    "round": server_round,
                    "phase": "during",
                    "t": round(time.time() - t0, 3),

                    # YARN memory
                    "total_mb": s["total_mb"],
                    "allocated_mb": s["allocated_mb"],
                    "free_mb": s["free_mb"],

                    # YARN vcores
                    "total_vcores": s["total_vcores"],
                    "allocated_vcores": s["allocated_vcores"],
                    "free_vcores": s["free_vcores"],

                    # Spark config
                    "spark_executors": num_exec,
                    "spark_executor_cores": self.cores_per_exec,
                    "spark_executor_mem_gb": mem_per_exec_gb,
                })

                time.sleep(0.5)

            # Wait for process output (and final return code)
            stdout, stderr = p.communicate()
            agg_seconds = round(time.time() - t0, 3)

            # Save Spark logs (very useful)
            (Path("./logs") / f"spark_round_{server_round}.stdout.txt").write_text(stdout or "")
            (Path("./logs") / f"spark_round_{server_round}.stderr.txt").write_text(stderr or "")

            # Print tail for quick debug
            print("\n[Server] Spark STDOUT (last 60 lines):\n", "\n".join((stdout or "").splitlines()[-60:]))
            print("\n[Server] Spark STDERR (last 60 lines):\n", "\n".join((stderr or "").splitlines()[-60:]))

            # Fail fast if Spark failed
            if p.returncode != 0:
                raise RuntimeError("Spark aggregation failed")

            # -----------------------------
            # AFTER Spark job
            # -----------------------------
            free_after = self.rm.free_resources()

            append_csv(log_path, {
                "round": server_round,
                "phase": "after",
                "t": agg_seconds,

                # YARN memory
                "total_mb": free_after["total_mb"],
                "allocated_mb": free_after["allocated_mb"],
                "free_mb": free_after["free_mb"],

                # YARN vcores
                "total_vcores": free_after["total_vcores"],
                "allocated_vcores": free_after["allocated_vcores"],
                "free_vcores": free_after["free_vcores"],

                # Spark config + result
                "spark_executors": num_exec,
                "spark_executor_cores": self.cores_per_exec,
                "spark_executor_mem_gb": mem_per_exec_gb,
                "agg_seconds": agg_seconds,
                "num_clients_used": len(results),
            })



            # -----------------------------
            # Download aggregated model
            # -----------------------------
            local_out = td_path / f"global_round_{server_round}.npz"
            if not hdfs_exists(hdfs_out):
                raise RuntimeError(f"Aggregated model not found in HDFS: {hdfs_out}")

            hdfs_get(hdfs_out, str(local_out))

            with np.load(local_out, allow_pickle=False) as data:
                avg_nds = [data[k] for k in sorted(data.files)]

                new_params = ndarrays_to_params(avg_nds)
                metrics = {
                    "spark_executors": num_exec,
                    "spark_executor_cores": self.cores_per_exec,
                    "spark_executor_mem_gb": mem_per_exec_gb,
                    "agg_seconds": agg_seconds,
                    "free_vcores": free_vcores,
                    "free_mb": free_mb,
                    "num_clients_used": len(results),
                }
                return new_params, metrics


    def configure_evaluate(self, server_round, parameters, client_manager):
        # Optional evaluation
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm-url", required=True, help="YARN ResourceManager base URL, e.g. http://rm-host:8088")
    ap.add_argument("--hdfs-base", default="hdfs:///fedl")
    ap.add_argument("--min-clients", type=int, default=2)
    ap.add_argument("--cores-per-exec", type=int, default=2)
    ap.add_argument("--max-exec", type=int, default=20)
    ap.add_argument("--min-exec", type=int, default=2)
    ap.add_argument("--exec-mem-cap", type=int, default=8)
    ap.add_argument("--spark-submit", default="spark-submit")
    ap.add_argument("--spark-agg-script", default="spark_agg.py")
    ap.add_argument("--bind", default="0.0.0.0:8080")
    args = ap.parse_args()

    strategy = ResourceAwareSparkFedAvg(
        rm_url=args.rm_url,
        hdfs_base=args.hdfs_base,
        min_clients=args.min_clients,
        cores_per_exec=args.cores_per_exec,
        max_exec=args.max_exec,
        min_exec=args.min_exec,
        exec_mem_gb_cap=args.exec_mem_cap,
        spark_submit=args.spark_submit,
        spark_agg_script=args.spark_agg_script,
    )

    fl.server.start_server(
        server_address=args.bind,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
