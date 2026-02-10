# Resource-Aware Federated Learning on YARN

This repository implements a resource-aware federated learning (FL) system that integrates Flower, Apache Spark, HDFS, and YARN to perform scalable and adaptive model aggregation on shared cluster infrastructure.

The system dynamically adapts Spark aggregation resources (executors, memory, and cores) based on real-time YARN cluster availability, enabling federated learning workloads to coexist efficiently with other 

## Requirements
1. Install [Apache hadoop](https://hadoop.apache.org/releases.html)
2. Install required python packages.
```bash
pip install requirements.txt
```

## Usage

1. Start HDFS and YARN services.
2. Launch the Flower server.
```
python server.py --rm-url http://<rm-host>:8088
```
3. Start multiple FL clients:
```
python client.py --cid <client_id>
```

4. After training, analyze logs and plots:
```
python plot_yarn_resource_awareness.py
```

5. Evaluate the final global model:
```
python evaluate.py
```

## System Overview

The system consists of the following components:

- **Federated Clients :**
Train local models on private data partitions and send model updates to the FL server.
- **Flower Server (Coordinator) :**
Orchestrates FL rounds, serializes client updates, queries YARN cluster metrics, launches Spark aggregation jobs, and distributes updated global models.
- **Apache Spark on YARN :**
Performs distributed weighted aggregation of model parameters using cluster resources determined at runtime.
- **HDFS :**
Stores client updates, aggregation manifests, and global models for reliable inter-process exchange.
- **YARN ResourceManager :**
Provides real-time cluster resource metrics via REST API, enabling adaptive resource allocation.

## Workflow

Clients train locally and send model updates to the Flower server.

- The server serializes updates into .npz files and uploads them to HDFS.

- The server queries YARN for current cluster resources.

- A Spark aggregation job is launched with dynamically computed resources.

- Spark reads updates from HDFS, performs weighted aggregation, and writes the global model back to HDFS.

- Resource usage is logged throughout the aggregation lifecycle.

The next federated learning round begins.
