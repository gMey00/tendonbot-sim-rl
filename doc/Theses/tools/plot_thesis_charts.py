import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".config"))
import plot_config as pcfg

pcfg.apply_style()

def plot_stacked_3v(out_path, title, csv_files, labels, colors):
    plt.figure()
    for csv_file, label, color in zip(csv_files, labels, colors):
        if not Path(csv_file).exists():
            print(f"Missing {csv_file}")
            continue
        df = pd.read_csv(csv_file)
        plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=label, color=color, linewidth=1.0)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    pcfg.finalize(plt.gcf(), out_path, title=title, tight=True)

