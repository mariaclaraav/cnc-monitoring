import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple, Dict
from src.utils.frequency.get_frequency import FrequencyProcessor

class FrequencyAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        column: str = 'Y_axis',
        sampling_rate: int = 2000,
        height_thresh: float = 0.05,
        plot: bool = False
    ):
        """
        Initializes the FrequencyAnalyzer with necessary parameters.

        Parameters:
        - df (pd.DataFrame): DataFrame containing 'Unique_Code' and the specified column.
        - column (str): Column name to analyze. Default is 'Y_axis'.
        - sampling_rate (int): Data sampling rate. Default is 2000.
        - height_thresh (float): Height threshold for high energy. Default is 0.05.
        - plot (bool): If True, generates plots. Default is False.
        """
        self.df = df.copy()
        self.column = column
        self.sampling_rate = sampling_rate
        self.height_thresh = height_thresh
        self.plot = plot

        # Initial analysis results
        self.high_energy_bins: Dict = {}
        self.high_density_bins: Dict = {}
        self.all_high_energy: List[Tuple[float, float, float]] = []
        self.energy_counter: Counter = Counter()
        self.common_energy_intervals: List[Tuple[Tuple[float, float], int]] = []
        self.top_intervals: List[Tuple[Tuple[float, float], int]] = []
        self.merged_intervals: List[Tuple[Tuple[float, float], int]] = []
        self.sorted_stats: List[Dict] = []
        self.bin_width: int = None

        # Refined analysis results
        self.refined_energy_bins: Dict = {}
        self.refined_density_bins: Dict = {}
        self.all_refined_energy: List[Tuple[float, float, float]] = []
        self.filtered_refined_energy: List[Tuple[float, float, float]] = []
        self.refined_data: List[Dict] = []
        self.top_occurrences_refined: Dict[str, List[Dict]] = {}
        self.top_std_refined: Dict[str, List[Dict]] = {}

    def _merge_intervals(self, intervals: List[Tuple[Tuple[float, float], int]]) -> List[Tuple[Tuple[float, float], int]]:
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda x: x[0][0])
        merged = []
        current_start, current_end = sorted_intervals[0][0]
        current_count = sorted_intervals[0][1]

        for (start, end), count in sorted_intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
                current_count += count
            else:
                merged.append(((current_start, current_end), current_count))
                current_start, current_end = start, end
                current_count = count
        merged.append(((current_start, current_end), current_count))
        return merged

    def _plot_initial(self, top_n: int, bin_width: int):
        labels = [f"{d['start_freq']:.0f}-{d['end_freq']:.0f} Hz" for d in self.sorted_stats]
        centers = [(d['start_freq'] + d['end_freq']) / 2 for d in self.sorted_stats]
        counts = [d['count'] for d in self.sorted_stats]
        energies = [d['energies'] for d in self.sorted_stats]

        plt.figure(figsize=(12, 6))
        plt.bar(centers, counts, width=bin_width * 0.8, color='blue', edgecolor='black')
        plt.xlabel('Frequency Interval (Hz)')
        plt.ylabel('Occurrences')
        plt.title(f"Top {top_n} Frequency Intervals with High Energy (Merged)")
        plt.xticks(centers, labels, rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        boxprops = dict(facecolor='blue', color='black')
        medianprops = dict(color='orange')

        plt.figure(figsize=(12, 6))
        plt.boxplot(
            energies,
            labels=labels,
            patch_artist=True,
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
        )
        plt.xlabel('Frequency Interval (Hz)')
        plt.ylabel('Max Energy')
        plt.title('Energy Distribution per Frequency Interval (Merged)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def analyze_frequency(self, bin_width: int = 50, top_n: int = 10):
        self.bin_width = bin_width
        grouped = self.df.groupby('Unique_Code')[[self.column, 'Unique_Code']]
        for code, group in grouped:
            merged_energy, merged_density = FrequencyProcessor(
                data=group,
                column_name=self.column,
                sampling_rate=self.sampling_rate,
                bin_width=bin_width,
                height_threshold=self.height_thresh,
                plot=False,
                print_values=False
            ).run()
            self.high_energy_bins[code] = merged_energy
            self.high_density_bins[code] = merged_density

        self.all_high_energy = [
            interval for intervals in self.high_energy_bins.values() for interval in intervals
        ]
        self.energy_counter = Counter((s, e) for s, e, _ in self.all_high_energy)
        self.common_energy_intervals = self.energy_counter.most_common()
        self.top_intervals = self.common_energy_intervals[:top_n]
        self.merged_intervals = self._merge_intervals(self.top_intervals)

        print(f"\nTop {top_n} frequency intervals with high energy (merged):")
        self.sorted_stats = []

        for (start, end), count in self.merged_intervals:
            energies = [
                energy for s, e, energy in self.all_high_energy
                if not (e <= start or s >= end)
            ]
            max_e = max(energies) if energies else 0
            mean_e = np.mean(energies) if energies else 0
            std_e = np.std(energies) if energies else 0

            self.sorted_stats.append({
                'start_freq': start,
                'end_freq': end,
                'count': count,
                'max_energy': max_e,
                'mean_energy': mean_e,
                'std_energy': std_e,
                'energies': energies
            })

            print(f"Frequency: {start:.2f} Hz - {end:.2f} Hz | "
                  f"Occurrences: {count} | Mean Energy: {mean_e:.2f} | "
                  f"Max Energy: {max_e:.2f} (Std: {std_e:.2f})")

        self.sorted_stats.sort(key=lambda x: x['start_freq'])

        if self.plot:
            self._plot_initial(top_n, bin_width)

        return self.sorted_stats

    def _plot_refined(self, global_max_std: float, global_max_occ: float):
        for data in self.refined_data:
            bins = data['refined_bins']
            label = data['interval_label']
            centers = [(b['start_freq'] + b['end_freq']) / 2 for b in bins]
            occurrences = [b['occurrences'] for b in bins]
            stds = [b['std_energy'] for b in bins]

            plt.figure(figsize=(12, 6))
            ax1 = plt.gca()
            ax1.bar(centers, occurrences, width=self.bin_width_refined * 0.8, color='blue', alpha=0.8, label='Occurrences')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Occurrences', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(0, global_max_occ + 10)

            ax2 = ax1.twinx()
            ax2.plot(centers, stds, 'r-o', label='Std Dev Energy')
            ax2.set_ylabel('Std Dev Energy', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, global_max_std + 10)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            plt.title(f"Occurrences and Std Dev of Energy in Interval {label} (Refined)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def refine_frequency_analysis(self, bin_width_refined: int = 10, top_n_refined: int = 3):
        if self.bin_width is None:
            raise ValueError("Initial frequency analysis not performed. Call 'analyze_frequency' first.")
        if bin_width_refined >= self.bin_width:
            raise ValueError(f"Refined bin_width ({bin_width_refined} Hz) must be less than initial bin_width ({self.bin_width} Hz).")
        if not self.merged_intervals:
            raise ValueError("No merged intervals found. Run 'analyze_frequency' first.")

        self.bin_width_refined = bin_width_refined
        self.top_occurrences_refined = { 
            f"{s:.0f}-{e:.0f} Hz": [] for (s, e), _ in self.merged_intervals
        }
        self.top_std_refined = { 
            f"{s:.0f}-{e:.0f} Hz": [] for (s, e), _ in self.merged_intervals
        }

        grouped = self.df.groupby('Unique_Code')[[self.column, 'Unique_Code']]
        for code, group in grouped:
            
            
            merged_energy_refined, merged_density_refined = FrequencyProcessor(
                data=group,
                column_name=self.column,
                sampling_rate=self.sampling_rate,
                bin_width=bin_width_refined,
                height_threshold=self.height_thresh,
                plot=False,
                print_values=False
            ).run()
            self.refined_energy_bins[code] = merged_energy_refined
            self.refined_density_bins[code] = merged_density_refined

        self.all_refined_energy = [
            interval for intervals in self.refined_energy_bins.values() for interval in intervals
        ]

        for s_ref, e_ref, energy in self.all_refined_energy:
            for (s_top, e_top), _ in self.merged_intervals:
                if s_ref >= s_top and e_ref <= e_top:
                    self.filtered_refined_energy.append((s_ref, e_ref, energy))
                    break

        all_std = []
        all_occ = []

        for (s_top, e_top), _ in self.merged_intervals:
            label = f"{s_top:.0f}-{e_top:.0f} Hz"
            bins = np.arange(s_top, e_top + bin_width_refined, bin_width_refined)
            if len(bins) < 2:
                print(f"Interval {label} too small to subdivide.")
                continue

            refined_bins = []
            for i in range(len(bins) - 1):
                b_start, b_end = bins[i], bins[i + 1]
                energies = [
                    energy for s, e, energy in self.filtered_refined_energy
                    if not (e <= b_start or s >= b_end)
                ]
                occ = len(energies)
                std_e = np.std(energies) if energies else 0

                refined_bins.append({
                    'start_freq': b_start,
                    'end_freq': b_end,
                    'occurrences': occ,
                    'std_energy': std_e,
                    'interval_label': label
                })

                all_std.append(std_e)
                all_occ.append(occ)

            self.refined_data.append({
                'refined_bins': refined_bins,
                'interval_label': label
            })

            # Filtrar bins com std_energy > 0 antes de selecionar os top_n_refined
            top_occ = sorted(refined_bins, key=lambda x: x['occurrences'], reverse=True)
            top_occ_filtered = [bin_data for bin_data in top_occ if bin_data['std_energy'] > 0]
            self.top_occurrences_refined[label] = top_occ_filtered[:top_n_refined]

            top_std = sorted(refined_bins, key=lambda x: x['std_energy'], reverse=True)
            top_std_filtered = [bin_data for bin_data in top_std if bin_data['std_energy'] > 0]
            self.top_std_refined[label] = top_std_filtered[:top_n_refined]

        if self.plot:
            global_max_std = max(all_std) if all_std else 1
            global_max_occ = max(all_occ) if all_occ else 1
            self._plot_refined(global_max_std, global_max_occ)

        return self.refined_data

    def display_top_intervals(self, top_n: int):
        print(f"\nTop {top_n} Refined Intervals by Occurrences:")
        for label, tops in self.top_occurrences_refined.items():
            print(f"\nInitial Interval: {label}")
            for idx, bin_data in enumerate(tops, 1):
                print(f"  {idx}. Frequency: {bin_data['start_freq']:.2f} Hz - {bin_data['end_freq']:.2f} Hz | "
                      f"Occurrences: {bin_data['occurrences']} | Std Dev: {bin_data['std_energy']:.2f}")

        print(f"\nTop {top_n} Refined Intervals by Std Dev:")
        for label, tops in self.top_std_refined.items():
            print(f"\nInitial Interval: {label}")
            for idx, bin_data in enumerate(tops, 1):
                print(f"  {idx}. Frequency: {bin_data['start_freq']:.2f} Hz - {bin_data['end_freq']:.2f} Hz | "
                      f"Occurrences: {bin_data['occurrences']} | Std Dev: {bin_data['std_energy']:.2f}")

    def run_analysis(
        self, 
        top_n_initial: int = 10, 
        top_n_refined: int = 3, 
        bin_width: int = 50, 
        bin_width_refined: int = 10
    ):
        """
        Executes initial and refined frequency analysis, and displays top refined intervals.

        Parameters:
        - top_n_initial (int): Number of top initial intervals. Default is 10.
        - top_n_refined (int): Number of top refined intervals per initial interval. Default is 3.
        - bin_width (int): Bin width for initial analysis. Default is 50.
        - bin_width_refined (int): Bin width for refined analysis. Default is 10.

        Returns:
        - Dict containing top refined occurrences and std deviations.
        """
        self.analyze_frequency(bin_width=bin_width, top_n=top_n_initial)
        self.refine_frequency_analysis(bin_width_refined=bin_width_refined, top_n_refined=top_n_refined)
        self.display_top_intervals(top_n_refined)
        
        return {
            'top_occurrences_refined': self.top_occurrences_refined,
            'top_std_refined': self.top_std_refined
        }
