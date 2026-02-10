import requests

class YarnMonitor:
    """
    Reads cluster metrics from YARN ResourceManager REST API.
    """
    def __init__(self, rm_base_url: str, timeout: float = 5.0):
        self.rm = rm_base_url.rstrip("/")
        self.timeout = timeout

    def cluster_metrics(self) -> dict:
        url = f"{self.rm}/ws/v1/cluster/metrics"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["clusterMetrics"]

    def free_resources(self) -> dict:
        m = self.cluster_metrics()
        # These fields exist in most YARN versions
        total_mb = int(m.get("totalMB", 0))
        allocated_mb = int(m.get("allocatedMB", 0))
        total_vcores = int(m.get("totalVirtualCores", 0))
        allocated_vcores = int(m.get("allocatedVirtualCores", 0))

        return {
            "total_mb": total_mb,
            "allocated_mb": allocated_mb,
            "free_mb": max(0, total_mb - allocated_mb),
            "total_vcores": total_vcores,
            "allocated_vcores": allocated_vcores,
            "free_vcores": max(0, total_vcores - allocated_vcores),
        }
