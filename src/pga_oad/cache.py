from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class DataGolfCache:
    DEFAULT_TTL = 3600          # 1 hour for live endpoints
    HISTORICAL_TTL = 86400      # 24 hours for historical endpoints

    HISTORICAL_PREFIXES = (
        "historical-raw-data",
        "historical-event-data",
        "historical-odds",
        "historical-dfs-data",
    )

    def __init__(self, cache_dir: str | Path = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, endpoint: str, params: dict) -> Path:
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        safe_endpoint = endpoint.strip("/").replace("/", "_")
        return self.cache_dir / f"{safe_endpoint}_{params_hash}.json"

    def _ttl_for(self, endpoint: str) -> int:
        for prefix in self.HISTORICAL_PREFIXES:
            if prefix in endpoint:
                return self.HISTORICAL_TTL
        return self.DEFAULT_TTL

    def get(self, endpoint: str, params: dict) -> Optional[Any]:
        path = self._key_to_path(endpoint, params)
        if not path.exists():
            return None
        with path.open() as f:
            entry = json.load(f)
        ttl = self._ttl_for(endpoint)
        if time.time() - entry["timestamp"] > ttl:
            path.unlink(missing_ok=True)
            return None
        return entry["data"]

    def set(self, endpoint: str, params: dict, data: Any) -> None:
        path = self._key_to_path(endpoint, params)
        with path.open("w") as f:
            json.dump({"timestamp": time.time(), "data": data}, f)
