import argparse
import json
from io import BytesIO
import numpy as np
from pyspark.sql import SparkSession

def load_npz_bytes(b: bytes) -> list[np.ndarray]:
    with np.load(BytesIO(b), allow_pickle=False) as data:
        return [data[k] for k in sorted(data.files)]

def assert_npz_bytes(b: bytes, path: str):
    if len(b) < 4:
        raise RuntimeError(f"Empty/truncated file from HDFS: {path}, size={len(b)}")
    if not (b.startswith(b"PK\x03\x04") or b.startswith(b"PK\x05\x06") or b.startswith(b"PK\x07\x08")):
        raise RuntimeError(f"Not a valid NPZ/ZIP from HDFS: {path}, first4={b[:4]!r}, size={len(b)}")

def read_hdfs_bytes_driver(sc, path: str) -> bytes:
    jvm = sc._jvm
    hconf = sc._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)
    p = jvm.org.apache.hadoop.fs.Path(path)

    stream = fs.open(p)
    baos = jvm.java.io.ByteArrayOutputStream()
    try:
        jvm.org.apache.hadoop.io.IOUtils.copyBytes(stream, baos, hconf, False)
        return bytes(baos.toByteArray())
    finally:
        stream.close()
        baos.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    spark = SparkSession.builder.appName(f"fedl-agg-round-{args.round}").getOrCreate()
    sc = spark.sparkContext

    manifest_text = "\n".join(sc.textFile(args.manifest).collect())
    manifest = json.loads(manifest_text)

    loaded = []
    for item in manifest:
        path = item["path"]
        n = int(item["num_examples"])
        b = read_hdfs_bytes_driver(sc, path)
        assert_npz_bytes(b, path)
        arrays = load_npz_bytes(b)
        loaded.append((n, arrays))

    if not loaded:
        raise RuntimeError("Manifest empty: no client updates")

    rdd = sc.parallelize(loaded, len(loaded))

    def seq_op(acc, value):
        n, arrays = value
        if acc is None:
            return (n, [a.astype(np.float64) * n for a in arrays])
        total_n, sum_arrays = acc
        for i in range(len(sum_arrays)):
            sum_arrays[i] += arrays[i].astype(np.float64) * n
        return (total_n + n, sum_arrays)

    def comb_op(a, b):
        if a is None: return b
        if b is None: return a
        total_a, sum_a = a
        total_b, sum_b = b
        for i in range(len(sum_a)):
            sum_a[i] += sum_b[i]
        return (total_a + total_b, sum_a)

    total_n, sum_arrays = rdd.aggregate(None, seq_op, comb_op)
    avg_arrays = [(s / total_n).astype(np.float32) for s in sum_arrays]

    out_bytes = BytesIO()
    np.savez(out_bytes, *avg_arrays)
    out_b = out_bytes.getvalue()

    jvm = sc._jvm
    hconf = sc._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)
    out_path = jvm.org.apache.hadoop.fs.Path(args.out)

    if fs.exists(out_path):
        fs.delete(out_path, True)

    stream = fs.create(out_path, True)
    stream.write(bytearray(out_b))
    stream.close()

    spark.stop()

if __name__ == "__main__":
    main()
