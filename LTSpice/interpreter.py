"""Author: Nikola D. Lilov, 24.VII.2026
  
   Desc: Interpreter for LTspice .raw files en masse. Before this program,
   one would have run generator.exe on an existing .asc schematic in order
   to create many .asc variations and their corresponding .raw files. This
   program will then read all of those .raw files and visualize the data.
  
   Usage: python interpreter.py <path_to_raw_files_dir> <template_basename> [output_dir]
   Usage: interpreter.exe <path_to_raw_files_dir> <template_basename> [output_dir]
"""


import sys
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from PyLTSpice import RawRead


DISTINCT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def natural_key(path: Path):
    """Split filename into text/number chunks so template_2 sorts before template_10."""
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r'(\d+)', path.stem)]


def find_raw_files(directory: Path, template: str):
    all_raw = directory.glob(f"{template}_*.raw")
    # Exclude operating-point raw files (name ends in ".op.raw")
    data_raw = [f for f in all_raw if not f.name.lower().endswith(".op.raw")]
    return sorted(data_raw, key=natural_key)


def main():
    if len(sys.argv) not in (3, 4):
        print(f"\nUsage: {sys.argv[0]} <path_to_raw_files_dir> <template_basename> [output_dir]\n")
        sys.exit(1)

    directory = Path(sys.argv[1])
    template = sys.argv[2]
    out_dir = Path(sys.argv[3]) if len(sys.argv) == 4 else directory / "_plots"
    if out_dir.exists():
        answer = input(f"\nWarning: The directory {out_dir} already exists. Overwrite it? (y/n) ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborting.\n")
            sys.exit(1)
        shutil.rmtree(out_dir)  # wipe existing contents
    out_dir.mkdir(parents=True)

    files = find_raw_files(directory, template)
    if not files:
        print(f"No matching .raw files found for pattern '{template}_*.raw' in {directory}")
        sys.exit(1)

    print(f"\nFound {len(files)} .raw files.")

    # --- Choose traces once, based on the first file ---
    first_raw = RawRead(str(files[0]))
    traces = first_raw.get_trace_names()
    x_axis_name = traces[0]  # "time" for .tran, "frequency" for .ac, etc.

    print(f"X axis will be: {x_axis_name}\n")
    print("Choose which traces to plot:")
    for i, trace in enumerate(traces[1:]):
        print(f"{i+1}: {trace}", end="\t")
        if (i + 1) % 5 == 0:
            print()
    print()
    input_traces = input("Enter the wanted trace indexes, separated by spaces: ")
    wanted_traces = [traces[int(i)] for i in input_traces.split()]

#    print("\nConfigure Y axes:")
#    print("  - Leave blank to auto-assign current-like traces to the secondary axis.")
#    print("  - Or enter the selected trace numbers to place on the secondary axis.")
#    for i, trace in enumerate(wanted_traces, start=1):
#        print(f"  {i}: {trace}")
#    secondary_input = input("Secondary axis trace indexes (blank = auto): ").strip()

    def is_current_trace(name: str) -> bool:
        return name.strip().startswith("I")

#    if secondary_input:
#        try:
#            secondary_indexes = {int(tok) for tok in secondary_input.split()}
#        except ValueError:
#            secondary_indexes = set()
#        right_traces = [trace for i, trace in enumerate(wanted_traces, start=1)
#                        if i in secondary_indexes]
#    else:
    right_traces = [trace for trace in wanted_traces if is_current_trace(trace)]

    left_traces = [trace for trace in wanted_traces if trace not in right_traces]
#    if right_traces:
#        print(f"Using secondary Y axis for: {', '.join(right_traces)}")
#    else:
#        print("Using a single Y axis for all traces.")

    # --- First pass: find global axis limits so every plot is comparable ---
    # Note: this caches every RawRead in memory for a second pass below.
    # For very large sweeps (thousands of files / huge waveforms), swap this
    # for a two-pass approach that re-reads each file instead of caching.
    print("\nScanning all files to compute shared axis limits...")
    x_min, x_max = float("inf"), float("-inf")
    left_min, left_max = float("inf"), float("-inf")
    right_min, right_max = float("inf"), float("-inf")

    raws = {}
    for f in files:
        r = RawRead(str(f))
        raws[f] = r
        x = r.get_trace(x_axis_name).get_wave()
        x_min, x_max = min(x_min, x.min()), max(x_max, x.max())
        for trace in left_traces:
            y = r.get_trace(trace).get_wave()
            left_min, left_max = min(left_min, y.min()), max(left_max, y.max())
        for trace in right_traces:
            y = r.get_trace(trace).get_wave()
            right_min, right_max = min(right_min, y.min()), max(right_max, y.max())

    # small padding so curves aren't glued to the frame edges
    left_pad = 0.05 * (left_max - left_min if left_max > left_min else 1.0)
    left_min, left_max = left_min - left_pad, left_max + left_pad
    if right_traces:
        right_pad = 0.05 * (right_max - right_min if right_max > right_min else 1.0)
        right_min, right_max = right_min - right_pad, right_max + right_pad

    print("Visualizing...")
    # --- Second pass: plot each file with the same axes ---
    for f in files:
        r = raws[f]
        x = r.get_trace(x_axis_name).get_wave()

        fig, ax = plt.subplots()
        lines = []
        labels = []

        for idx, trace in enumerate(left_traces):
            y = r.get_trace(trace).get_wave()
            color = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]
            line, = ax.plot(x, y, label=trace, color=color)
            lines.append(line)
            labels.append(trace)

        if right_traces:
            ax2 = ax.twinx()
            for idx, trace in enumerate(right_traces):
                y = r.get_trace(trace).get_wave()
                color = DISTINCT_COLORS[(len(left_traces) + idx) % len(DISTINCT_COLORS)]
                line, = ax2.plot(x, y, label=trace, color=color)
                lines.append(line)
                labels.append(trace)
            ax2.set_ylim(right_min, right_max)
            ax2.set_ylabel("Secondary axis value")
        else:
            ax2 = None

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(left_min, left_max)
        ax.set_xlabel(x_axis_name)
        ax.set_ylabel("Primary axis value")
        ax.set_title(f.stem)
        ax.legend(lines, labels)
        ax.grid(True)

        out_path = out_dir / f"{f.stem}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"\nDone. {len(files)} PNGs saved to {out_dir}\n")


if __name__ == "__main__":
    main()
