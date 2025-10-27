from pathlib import Path
import json

if __name__ == '__main__':
    # --- Paths ---
    base = Path(r"C:\Users\Administrator\Desktop\digitization-pid")
    data_dir = base / r"src\data\dummy"
    out_dir = base / r"app\detections"

    # Ensure output folder exists
    out_dir.mkdir(parents=True, exist_ok=True)

    values_path = data_dir / "values.json"
    thresholds_path = data_dir / "threshold.json"
    out_path = out_dir / "anomalies.json"

    # --- Load data ---
    with open(values_path, "r", encoding="utf-8") as f:
        values = json.load(f)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    readings = values.get("pressure_readings", [])
    threshold_map = thresholds.get("pressure_thresholds", {})

    # --- Detect anomalies ---
    anomalies = []
    for row in readings:
        ts = row.get("timestamp")
        for comp, val in row.items():
            if comp == "timestamp":
                continue
            thr = threshold_map.get(comp)
            if thr is None:
                continue

            try:
                v = float(val)
            except (TypeError, ValueError):
                anomalies.append({
                    "component": comp,
                    "observed_value": val,
                    "timestamp": ts,
                    "expected_range": {
                        "min": thr.get("min"),
                        "max": thr.get("max"),
                        "units": thr.get("units")
                    }
                })
                continue

            min_v = thr.get("min")
            max_v = thr.get("max")

            if min_v is None or max_v is None:
                continue

            if not (min_v <= v <= max_v):
                anomalies.append({
                    "component": comp,
                    "observed_value": v,
                    "timestamp": ts,
                    "expected_range": {
                        "min": min_v,
                        "max": max_v,
                        "units": thr.get("units")
                    }
                })

    # --- Save anomalies ---
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"anomalies": anomalies}, f, indent=2)

    print(f"Anomalies found: {len(anomalies)}")
    print(f"Saved to: {out_path}")
    if anomalies[:3]:
        print("First few:", json.dumps(anomalies[:3], indent=2))
