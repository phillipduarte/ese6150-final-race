import csv
import sys
import math

def convert_raceline(input_path: str, output_path: str):
    """
    Convert TUM raceline CSV format to x, y, yaw, velocity CSV.

    Input columns:  s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    Output columns: x, y, yaw, velocity
    """
    rows = []

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 6:
                continue
            x        = float(parts[1])
            y        = float(parts[2])
            yaw      = float(parts[3])   # psi_rad
            velocity = float(parts[5])   # vx_mps
            rows.append((x, y, yaw, velocity))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "yaw", "velocity"])
        writer.writerows(rows)

    print(f"Converted {len(rows)} waypoints -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_raceline.py <input.csv> <output.csv>")
        sys.exit(1)
    convert_raceline(sys.argv[1], sys.argv[2])