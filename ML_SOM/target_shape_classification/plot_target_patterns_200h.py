#!/usr/bin/env python3
"""Plot the four requested early-priority PCE pattern schematics."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def main():
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(parents=True, exist_ok=True)
    examples = [
        ('IFO-Bridge', '#3b8b69',
         [0, 35, 80, 140, 200, 350, 500, 650],
         [.61, .88, .99, 1.0, .99, .93, .87, .81],
         'rapid rise · broad plateau · slow decline'),
        ('IFO-Hill', '#c36437',
         [0, 35, 75, 110, 160, 200, 350, 500, 650],
         [.61, .78, 1.0, .90, .70, .62, .54, .49, .45],
         'rapid rise · sharp peak · fast then slow decline'),
        ('IFO-Slope', '#5a73ad',
         [0, 35, 80, 130, 200, 350, 500, 650],
         [1.0, .88, .73, .64, .59, .55, .51, .47],
         'fast initial decline · slow continuous decline'),
        ('IFO-Valley', '#b18a2c',
         [0, 35, 75, 110, 155, 200, 350, 500, 650],
         [1.0, .77, .44, .59, .82, .86, .82, .77, .71],
         'early minimum · recovery · slow decline'),
    ]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 7.6), sharex=True, sharey=True)
    for ax, (name, color, knots_t, knots_y, description) in zip(axs.ravel(), examples):
        t = np.linspace(0, 650, 651)
        y = PchipInterpolator(knots_t, knots_y)(t)
        ax.axvspan(0, 200, color='#e9f3e9', zorder=0)
        ax.axvspan(200, 650, color='#fff5e2', zorder=0)
        ax.axvline(200, color='#808080', lw=1, ls='--', zorder=1)
        ax.plot(t, y, color=color, lw=3, zorder=2)
        ax.set_title(name, fontweight='bold', fontsize=14, pad=12)
        ax.text(100, 1.045, 'Initial stage (0–200 h)', ha='center', va='top',
                fontsize=9, color='#345d44')
        ax.text(425, 1.045, 'Long-term stage (>200 h)', ha='center', va='top',
                fontsize=9, color='#80612c')
        ax.text(0.5, 0.04, description, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=8.5, color='#404040')
        ax.set_xlim(0, 650)
        ax.set_ylim(.28, 1.07)
        ax.set_xticks([0, 100, 200, 400, 600])
        ax.set_yticks([.4, .6, .8, 1.0])
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Output (normalized PCE)')
        ax.grid(alpha=.18, zorder=0)
    fig.suptitle('Four target PCE patterns: early behavior has priority',
                 fontsize=16, fontweight='bold', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .975), h_pad=2.1, w_pad=2.3)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(out / f'four_target_patterns_200h.{extension}', dpi=200,
                    bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
