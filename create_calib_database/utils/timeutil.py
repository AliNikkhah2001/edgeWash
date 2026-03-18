from datetime import datetime, timezone
def iso_to_ts(s: str) -> float: 
    return datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()
def ts_to_iso(t: float) -> str:  
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")
