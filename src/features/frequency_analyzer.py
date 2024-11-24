from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from src.utils.frequency.get_frequency import FrequencyProcessor

#REF:  Refactor the code, make it cleaner and more readable
class FrequencyAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        column: str = "Y_axis",
        sampling_rate: int = 2000,
        height_thresh: float = 0.05,
        C: float = 0.15,
        plot: bool = False,
    ):
        """
        Initializes the FrequencyAnalyzer with necessary parameters.

        Parameters:
        - df (pd.DataFrame): DataFrame containing 'Unique_Code' and the specified column.
        - column (str): Column name to analyze. Default is 'Y_axis'.
        - sampling_rate (int): Data sampling rate. Default is 2000.
        - height_thresh (float): Height threshold for high energy. Default is 0.05.
        - C (float): Coefficient for std energy threshold. Default is 0.2.
        - plot (bool): If True, generates plots. Default is False.
        """
        self.df = df.copy()
        self.column = column
        self.sampling_rate = sampling_rate
        self.height_thresh = height_thresh
        self.__C = C
        self.plot = plot

        # Analysis results
        self.high_energy_bins: Dict = {}
        self.all_high_energy: List[Tuple[float, float, float]] = []
        self.energy_counter: Counter = Counter()
        self.common_energy_intervals: List[Tuple[Tuple[float, float], int]] = []
        self.top_intervals: List[Tuple[Tuple[float, float], int]] = []
        self.merged_intervals: List[Tuple[Tuple[float, float], int]] = []
        self.sorted_stats: List[Dict] = []

        # Refined analysis results
        self.refined_energy_bins: Dict = {}
        self.all_refined_energy: List[Tuple[float, float, float]] = []
        self.filtered_refined_energy: List[Tuple[float, float, float]] = []
        self.refined_data: List[Dict] = []

    def _merge_intervals(
        self, intervals: List[Tuple[Tuple[float, float], int]]
    ) -> List[Tuple[Tuple[float, float], int]]:
        if not intervals:
            return []

        intervals.sort(key=lambda x: x[0][0])
        merged = [intervals[0]]

        for current in intervals[1:]:
            prev_start, prev_end = merged[-1][0]
            prev_count = merged[-1][1]
            start, end = current[0]

            if start <= prev_end:
                merged[-1] = ((prev_start, max(prev_end, end)), prev_count + current[1])
            else:
                merged.append(current)

        return merged

    def _plot_initial(self, top_n: int, bin_width: int):
        labels = [f"{d['start_freq']:.0f}-{d['end_freq']:.0f} Hz" for d in self.sorted_stats]
        centers = [(d["start_freq"] + d["end_freq"]) / 2 for d in self.sorted_stats]
        counts = [d["count"] for d in self.sorted_stats]
        energies = [d["energies"] for d in self.sorted_stats]

        plt.figure(figsize=(12, 6))
        plt.bar(centers, counts, width=bin_width * 0.8, color="blue", edgecolor="black")
        plt.xlabel("Frequency Interval (Hz)")
        plt.ylabel("Occurrences")
        plt.title(f"Top {top_n} Frequency Intervals with High Energy (Merged)")
        plt.xticks(centers, labels, rotation=45, ha="right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.boxplot(
            energies,
            labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor="blue", color="black"),
            medianprops=dict(color="orange"),
        )
        plt.xlabel("Frequency Interval (Hz)")
        plt.ylabel("Max Energy")
        plt.title("Energy Distribution per Frequency Interval (Merged)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def analyze_frequency(self, bin_width: int = 50, top_n: int = 10):
        grouped = self.df.groupby("Unique_Code")[[self.column, "Unique_Code"]]

        for code, group in grouped:
            merged_energy, _ = FrequencyProcessor(
                data=group,
                column_name=self.column,
                sampling_rate=self.sampling_rate,
                bin_width=bin_width,
                height_threshold=self.height_thresh,
                plot=False,
                print_values=False,
            ).run()
            self.high_energy_bins[code] = merged_energy

        self.all_high_energy = [
            interval for intervals in self.high_energy_bins.values() for interval in intervals
        ]
        self.energy_counter = Counter((s, e) for s, e, _ in self.all_high_energy)
        self.common_energy_intervals = self.energy_counter.most_common()
        self.top_intervals = self.common_energy_intervals[:top_n]
        self.merged_intervals = self._merge_intervals(self.top_intervals)

        self.sorted_stats = [
            {
                "start_freq": start,
                "end_freq": end,
                "count": count,
                "energies": [
                    energy
                    for s, e, energy in self.all_high_energy
                    if not (e <= start or s >= end)
                ],
            }
            for (start, end), count in self.merged_intervals
        ]

        if self.plot:
            self._plot_initial(top_n, bin_width)

        return self.sorted_stats
          
    def _plot_refined(self, global_max_std: float, global_max_occ: float, filter_intervals):
        for data in self.refined_data:
            bins = data["refined_bins"]
            label = data["interval_label"]
            centers = [(b["start_freq"] + b["end_freq"]) / 2 for b in bins]
            occurrences = [b["occurrences"] for b in bins]
            stds = [b["std_energy"] for b in bins]

            plt.figure(figsize=(12, 6))
            ax1 = plt.gca()
            ax1.bar(centers, occurrences, width=self.bin_width_refined * 0.8, color="blue", alpha=0.8, label="Occurrences")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Occurrences", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")
            ax1.set_ylim(0, global_max_occ + 10)

            ax2 = ax1.twinx()
            ax2.plot(centers, stds, "r-o", label="Std Dev Energy")
            ax2.set_ylabel("Std Dev Energy", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.set_ylim(0, global_max_std + 10)

            for start_freq, end_freq in filter_intervals:
                if start_freq >= centers[0] and end_freq <= (centers[-1] + 1):
                    ax1.axvspan(start_freq, end_freq, color="yellow", alpha=0.3, label="Filter Interval")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            plt.title(f"Occurrences and Std Dev of Energy in Interval {label} (Refined)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def _get_peaks(self):
        all_stds, all_centers, all_bins = [], [], []

        for data in self.refined_data:
            bins = data["refined_bins"]
            centers = [(b["start_freq"] + b["end_freq"]) / 2 for b in bins]
            stds = [b["std_energy"] for b in bins]

            all_stds.extend(stds)
            all_centers.extend(centers)
            all_bins.extend(bins)

        max_std_energy = max(all_stds) if all_stds else 0
        height = self.__C * max_std_energy
        peaks, _ = find_peaks(all_stds, height=height)

        peak_intervals = [(all_bins[p]["start_freq"], all_bins[p]["end_freq"]) for p in peaks]
        peak_intervals.sort()
        
        filter_intervals = []
        if peak_intervals:
            current_start, current_end = peak_intervals[0]
            for start, end in peak_intervals[1:]:
                if start <= current_end + 30:
                    current_end = max(current_end, end)
                else:
                    if current_end - current_start >= 10:
                        filter_intervals.append((current_start, current_end))
                    current_start, current_end = start, end
            if current_end - current_start >= 10:
                filter_intervals.append((current_start, current_end))

        return filter_intervals

    def refine_frequency_analysis(self, bin_width_refined: int = 1):
        if not self.merged_intervals:
            raise ValueError("Initial analysis not performed. Run 'analyze_frequency' first.")

        self.bin_width_refined = bin_width_refined
        grouped = self.df.groupby("Unique_Code")[[self.column, "Unique_Code"]]

        for code, group in grouped:
            merged_energy_refined, _ = FrequencyProcessor(
                data=group,
                column_name=self.column,
                sampling_rate=self.sampling_rate,
                bin_width=bin_width_refined,
                height_threshold=self.height_thresh,
                plot=False,
                print_values=False,
            ).run()
            self.refined_energy_bins[code] = merged_energy_refined

        self.all_refined_energy = [
            interval for intervals in self.refined_energy_bins.values() for interval in intervals
        ]

        self.filtered_refined_energy = [
            (s, e, energy)
            for s, e, energy in self.all_refined_energy
            if any(s >= s_top and e <= e_top for (s_top, e_top), _ in self.merged_intervals)
        ]

        all_std, all_occ = [], []

        for (s_top, e_top), _ in self.merged_intervals:
            label = f"{s_top:.0f}-{e_top:.0f} Hz"
            bins = np.arange(s_top, e_top + bin_width_refined, bin_width_refined)
            if len(bins) < 2:
                continue

            refined_bins = []
            for i in range(len(bins) - 1):
                b_start, b_end = bins[i], bins[i + 1]
                energies = [
                    energy for s, e, energy in self.filtered_refined_energy if not (e <= b_start or s >= b_end)
                ]
                refined_bins.append(
                    {
                        "start_freq": b_start,
                        "end_freq": b_end,
                        "occurrences": len(energies),
                        "std_energy": np.std(energies) if energies else 0,
                    }
                )
                all_std.extend([np.std(energies)] if energies else [])
                all_occ.extend([len(energies)] if energies else [])

            self.refined_data.append({"refined_bins": refined_bins, "interval_label": label})

        filter_intervals = self._get_peaks()

        if self.plot:
            global_max_std = max(all_std) if all_std else 1
            global_max_occ = max(all_occ) if all_occ else 1
            self._plot_refined(global_max_std, global_max_occ, filter_intervals)

        return filter_intervals

    def run_analysis(
        self, top_n_initial: int = 10, bin_width: int = 50, bin_width_refined: int = 1
    ):
        """Executes initial and refined frequency analysis, and displays top refined intervals.

        Parameters:
        - top_n_initial (int): Number of top initial intervals. Default is 10.
        - bin_width (int): Bin width for initial analysis. Default is 50.
        - bin_width_refined (int): Bin width for refined analysis. Default is 1.

        Returns:
        - List of filtered intervals.
        """
        self.analyze_frequency(bin_width=bin_width, top_n=top_n_initial)
        intervals = self.refine_frequency_analysis(bin_width_refined=bin_width_refined)
        return intervals