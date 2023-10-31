from __future__ import annotations

from typing import *
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
import scipy

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

pd.set_option("display.max_rows", None, "display.max_columns", None)


def get_data(
        data_keys,
        plot_name,
):
    """
    Reads the raw data. Datakeys is a list of tuples with files to read. Each tuple should contain:
        - FrameworkName: taco/spmm, taco/sddmm or rise
        - BenchmarkName: This is the operation for rise and the matrix for taco
        - methods: either list of methods or just '*'
        - n_Evaluations
        - doe_samples

    Returns a dict with (FrameworkName, BenchmarkName) as keys and (pandas dataframes, list of method) as outputs.
    Each data frame has columns (Evaluations, Means, Stds, Method). Which means that the different methods have been stacked horizontally.
    """
    return_dict = {}
    for key in data_keys:
        framework_folder = key["Framework"]
        benchmark_folder = key["Operation"]
        data_folder = os.path.join(BASE_FOLDER, framework_folder, benchmark_folder)
        if not os.path.isdir(data_folder):
    	    return {}
        # Get the header names for different frameworks
        if framework_folder in ["taco/spmm", "taco/sddmm", "ablation/spmm", "ablation/sddmm", "ablation/RF", "ablation/tmp", "ablation/permutation"]:
            value_header = "compute_time"
            feasibility_header = None
        elif framework_folder in ["rise", "ablation/rise"]:
            value_header = "runtime"
            feasibility_header = "Valid"

        # Get the methods
        methods = key["Method"]
        df = pd.DataFrame()
        val_df = pd.DataFrame()
        for method in methods:
            if "base" in method:
                feasibility_header = None
                
            method_folder = os.path.join(data_folder, method)
            if not os.path.isdir(method_folder):
            	continue
            files = [f for f in os.listdir(method_folder) if ".csv" in f]
            mdf = pd.DataFrame()
            for i, f in enumerate(files):
                file_data = pd.read_csv(os.path.join(method_folder, f))
                tmp_feasibility_header = feasibility_header
                if tmp_feasibility_header is None:
                    file_data["Feasible"] = [True] * file_data.shape[0]
                    tmp_feasibility_header = "Feasible"
                file_data = file_data[[value_header, tmp_feasibility_header]]
                file_data["Run"] = [i] * file_data.shape[0]
                file_data["Method"] = [method] * file_data.shape[0]
                file_data.columns = ["Value", "Feasible", "Run", "Method"]
                min_so_far = np.inf
                file_data["Min_Value"] = [0] * file_data.shape[0]
                for j in range(file_data.shape[0]):
                    if file_data.iloc[j, 1]:
                        min_so_far = min(min_so_far, file_data.iloc[j, 0])
                    file_data.iloc[j, 4] = (np.nan if min_so_far == np.inf else min_so_far)
                file_data["Evaluations"] = np.arange(file_data.shape[0]) + 1
                file_data = file_data.loc[file_data['Evaluations'] <= key["Max_Evaluations"]]
                mdf = pd.concat([mdf, file_data], axis=0)
            statistics_df = pd.DataFrame()
            for t in mdf["Evaluations"].unique():
                for m in mdf["Method"].unique():
                    tmp_df = mdf.loc[(mdf['Evaluations'] == t) & (mdf["Method"] == m)]
                    row = {"Evaluations": t,
                           "Method": m,
                           "Mean": np.nanmean(tmp_df["Min_Value"]),
                           "Std": np.nanstd(tmp_df["Min_Value"]),
                           }
                    statistics_df = statistics_df.append(row, ignore_index=True)

            df = pd.concat([df, statistics_df], ignore_index=True)
            val_df = pd.concat([val_df, mdf], ignore_index=True)
        # Get the default, expert and doe
        if plot_name == "rise_abl":
            with open(os.path.join(data_folder, "manual_configs", "default.csv")) as f:
                default = float(f.readline())
            with open(os.path.join(data_folder, "manual_configs", "expert.csv")) as f:
                expert = float(f.readline())
        else:
            times = pd.read_csv(os.path.join(data_folder, "times.csv"))
            expert = np.median(times.iloc[:, 2])
            default = np.median(times.iloc[:, 3])

        doe = key["Doe"]
        ind = np.arange(df.shape[0])
        df.reindex(ind)
        val_df = val_df[["Method", "Run", "Evaluations", "Min_Value"]]
        val_df.columns = ["Method", "Run", "Evaluations", "Value"]
        val_df.reindex(ind)
        return_dict[(framework_folder, benchmark_folder)] = {
            "Statistics_df": df,
            "Values_df": val_df,
            "Method": methods,
            "Default": default,
            "Expert": expert,
            "Doe": doe,
            "Max_Evaluations": key["Max_Evaluations"],
        }
    return return_dict


def make_geo(
        data: Dict[Tuple[str, str], Tuple[pd.DataFrame, List[str], float, float, int, int]],
        key_set: List[Tuple[str, str]],
        time_points: List[int],
        plot_name: str,
):
    plt.rcParams.update({
        "font.size": 18})
    plt.rcParams["figure.figsize"] = [5, 3]
    sns.set_theme()

    if plot_name == "rise":
        op_name = "Operation"
    else:
        op_name = "Matrix"
    merged_df = pd.DataFrame()
    for key in key_set:
        if not key in data:
    	    continue
        val_df = data[key]["Values_df"]
        stat_df = data[key]["Statistics_df"]
        if time_points[0] <= 1:
            actual_time_points = [math.floor(data[key]["Max_Evaluations"] / (3 - t)) for t in range(3)]
            val_df = val_df.loc[val_df["Evaluations"].isin(actual_time_points)]
            val_df.loc[val_df.Evaluations <= actual_time_points[0], "Evaluations"] = 0.33
            val_df.loc[(val_df.Evaluations <= actual_time_points[1]) & (val_df.Evaluations > 0.33), "Evaluations"] = 0.66
            val_df.loc[val_df.Evaluations > actual_time_points[1], "Evaluations"] = 1.0
            stat_df.loc[stat_df.Evaluations <= actual_time_points[0], "Evaluations"] = 0.33
            stat_df.loc[(stat_df.Evaluations <= actual_time_points[1]) & (stat_df.Evaluations > 0.33), "Evaluations"] = 0.66
            stat_df.loc[stat_df.Evaluations > actual_time_points[1], "Evaluations"] = 1.0
        else:
            val_df = val_df.loc[val_df["Evaluations"].isin(time_points)]

        val_df["imp_exp"] = val_df.apply(lambda row: data[key]["Expert"] / row["Value"], axis=1)
        val_df["imp_def"] = val_df.apply(lambda row: data[key]["Default"] / row["Value"], axis=1)
        val_df["imp_remb"] = val_df.apply(lambda row: np.nan, axis=1)  # remb_values[int(row["Evaluations"])]/row["Value"], axis=1)
        val_df[op_name] = [key[1]] * val_df.shape[0]
        merged_df = merged_df.append(val_df)

    for a, b in zip(user_specified_methods, labels):
        merged_df.replace(to_replace=a, value=b, inplace=True)

    if merged_df.empty:
        print("WARNING NO DATA for one ablation plot")
        return

    # Add geo Mean
    for t in merged_df.Evaluations.unique():
        for m in merged_df["Method"].unique():
            t_df = merged_df.loc[(merged_df['Evaluations'] == t) & (merged_df["Method"] == m)]
            row = {"Evaluations": t,
                   "Method": m,
                   op_name: "GeoMean",
                   "imp_exp": scipy.stats.gmean(t_df["imp_exp"]),
                   "imp_def": scipy.stats.gmean(t_df["imp_def"]),
                   "imp_remb": scipy.stats.gmean(t_df["imp_remb"])}
            merged_df = merged_df.append(row, ignore_index=True)

    fig, ax = plt.subplots()
    merged_df.columns = ["Method", "Run", "Number of Configuration Evaluations", "Value", "Avg. Performance relative to Expert", "imp_def", "imp_remb", op_name]
    merged_df_m = merged_df[merged_df[op_name] == "GeoMean"]
    fig.suptitle("Impact of Hidden Constraints")
    palette = sns.color_palette()
    colors = ([3, 4, 9] if plot_name == "rise_abl" else [0, 6, 2, 7])
    sns.barplot(data=merged_df_m, x="Number of Configuration Evaluations", hue="Method", y="Avg. Performance relative to Expert", ax=ax, errorbar='sd', err_kws={'color': 'grey', 'linewidth': 0.5}, palette=[palette[i] for i in colors])  # ])
    ax.axhline(1, linestyle="dashed", color="grey", linewidth=.8)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{'{:.1f}'.format(tick)}x" for tick in yticks])
    ax.set(ylabel=None)
    fig.text(0.0, 0.5, "Performance rel. to Expert", va='center', rotation='vertical', fontsize=12)
    ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.38),
        borderaxespad=0,
        ncol=2,
        fontsize=10,
        fancybox=True,
        shadow=True,
        facecolor="white"
    )
    plt.subplots_adjust(bottom=0.2, top=0.68)
    plt.savefig(f"plots/{'-'.join(key_set[0][0].split('/'))}_1.pdf", dpi=600, bbox_inches='tight')


def make_geo2(
        data: Dict[Tuple[str, str], Tuple[pd.DataFrame, List[str], float, float, int, int]],
        key_set: List[Tuple[str, str]],
        time_points: List[int],
        plot_name: str,
):
    plt.rcParams.update({
        "font.size": 18})

    plt.rcParams["figure.figsize"] = [5, 2.5]
    sns.set_theme()

    if plot_name == "rise":
        op_name = "Operation"
    else:
        op_name = "Matrix"
    merged_df = pd.DataFrame()
    for key in key_set:
        if not key in data:
    	    continue
        val_df = data[key]["Values_df"]
        stat_df = data[key]["Statistics_df"]
        if time_points[0] <= 1:
            actual_time_points = [math.floor(data[key]["Max_Evaluations"] / (3 - t)) for t in range(3)]
            val_df = val_df.loc[val_df["Evaluations"].isin(actual_time_points)]
            val_df.loc[val_df.Evaluations <= actual_time_points[0], "Evaluations"] = 0.33
            val_df.loc[(val_df.Evaluations <= actual_time_points[1]) & (val_df.Evaluations > 0.33), "Evaluations"] = 0.66
            val_df.loc[val_df.Evaluations > actual_time_points[1], "Evaluations"] = 1.0
            stat_df.loc[stat_df.Evaluations <= actual_time_points[0], "Evaluations"] = 0.33
            stat_df.loc[(stat_df.Evaluations <= actual_time_points[1]) & (stat_df.Evaluations > 0.33), "Evaluations"] = 0.66
            stat_df.loc[stat_df.Evaluations > actual_time_points[1], "Evaluations"] = 1.0
        else:
            val_df = val_df.loc[val_df["Evaluations"].isin(time_points)]

        val_df["imp_exp"] = val_df.apply(lambda row: data[key]["Expert"] / row["Value"], axis=1)
        val_df["imp_def"] = val_df.apply(lambda row: data[key]["Default"] / row["Value"], axis=1)
        val_df["imp_remb"] = val_df.apply(lambda row: np.nan, axis=1)
        val_df[op_name] = [key[1]] * val_df.shape[0]
        merged_df = merged_df.append(val_df)

    if merged_df.empty:
        print("WARNING NO DATA for one ablation plot")
        return


    for a, b in zip(user_specified_methods, labels):
        merged_df.replace(to_replace=a, value=b, inplace=True)

    # Add geo Mean
    for t in merged_df.Evaluations.unique():
        for m in merged_df["Method"].unique():
            t_df = merged_df.loc[(merged_df['Evaluations'] == t) & (merged_df["Method"] == m)]
            row = {"Evaluations": t,
                   "Method": m,
                   op_name: "GeoMean",
                   "imp_exp": scipy.stats.gmean(t_df["imp_exp"]),
                   "imp_def": scipy.stats.gmean(t_df["imp_def"]),
                   "imp_remb": scipy.stats.gmean(t_df["imp_remb"])}
            merged_df = merged_df.append(row, ignore_index=True)

    fig, ax = plt.subplots()
    merged_df.columns = ["Method", "Run", "Number of Configuration Evaluations", "Value", "Avg. Performance relative to Expert", "imp_def", "imp_remb", op_name]
    merged_df_m = merged_df[merged_df[op_name] == "GeoMean"]
    fig.suptitle("Ablation analysis on SpMM")
    palette2 = sns.color_palette() + sns.color_palette('dark')
    sns.barplot(data=merged_df_m, x="Number of Configuration Evaluations", hue="Method", y="Avg. Performance relative to Expert", ax=ax, errorbar='sd', err_kws={'color': 'grey', 'linewidth': 0.5}, palette=[palette2[i] for i in [0, 8, 9, 13, 14, 16, 17, 18]])
    ax.axhline(1, linestyle="dashed", color="grey", linewidth=.8)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{'{:.1f}'.format(tick)}x" for tick in yticks])
    ax.set_ylim([.7, 1.5])
    ax.set(ylabel=None)
    fig.text(0.0, 0.44, "Performance rel. to Expert", va='center', rotation='vertical', fontsize=12)
    ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.8),
        borderaxespad=0,
        ncol=2,
        fontsize=10,
        fancybox=True,
        shadow=True,
        facecolor="white"
    )
    plt.subplots_adjust(bottom=0.2, top=0.55)
    plt.savefig(f"plots/{'-'.join(key_set[0][0].split('/'))}_2.pdf", dpi=600, bbox_inches='tight')


if __name__ == "__main__":

    BASE_FOLDER = "results"

    plot_log = True
    SHOW_DOE = True
    EXPERT = True
    DEFAULT = True
    shadows = False
    cutoffs = None
    YLABEL = "Runtime [ms]"
    XLABEL = "Evaluation iterations"

    for plot_name, plot_type in zip(["taco_abl", "taco_abl2", "rise_abl"], ["type1", "type2", "type1"]):
        print(plot_name, plot_type)

        if plot_name == "taco_abl":
            INSETS = {
                0: ([10, 60], [0.25, 0.4, 0.6, 0.4], [3, 4]),
                1: ([10, 60], [0.25, 0.4, 0.6, 0.4], [3, 4]),
            }
            labels = [
                "BaCO",
                "BaCO--",
                "Ytopt (GP)",
                "RFs",
            ]
            user_specified_methods = [
                "bayesian_optimization",
                "simple",
                "ytopt",
                "RF",

            ]
            title = "TACO SpMM Abl"
            time_points = [20, 40, 60]

            tmp_data_sets: List[Tuple[str, str, List[str], int, int]] = [
                ("ablation/spmm", "email-Enron", user_specified_methods, 60, 7),
                ("ablation/spmm", "amazon0312", user_specified_methods, 60, 7),
                ("ablation/spmm", "filter3D", user_specified_methods, 60, 7),
            ]
            key_sets: List[Tuple[str, str]] = [
                ("ablation/spmm", "email-Enron"),
                ("ablation/spmm", "amazon0312"),
                ("ablation/spmm", "filter3D"),
            ]

        elif plot_name == "taco_abl2":
            INSETS = {
                0: ([10, 60], [0.25, 0.4, 0.6, 0.4], [3, 4]),
                1: ([10, 60], [0.25, 0.4, 0.6, 0.4], [3, 4]),
            }
            labels = [
                "BaCO (Spearman kernel)",
                "Kendall kernel",
                "Hamming kernel",
                "Naive kernel",
                "Without model priors",
                "Without transformations"
            ]
            user_specified_methods = [
                "bayesian_optimization",
                "kendall",
                "hamming",
                "naive",
                "nolsp",
                "nolog"
            ]
            title = "TACO SpMM Abl"
            time_points = [20, 40, 60]

            tmp_data_sets: List[Tuple[str, str, List[str], int, int]] = [
                ("ablation/spmm", "email-Enron", user_specified_methods, 60, 7),
                ("ablation/spmm", "amazon0312", user_specified_methods, 60, 7),
                ("ablation/spmm", "filter3D", user_specified_methods, 60, 7),
            ]
            key_sets: List[Tuple[str, str]] = [
                ("ablation/spmm", "email-Enron"),
                ("ablation/spmm", "amazon0312"),
                ("ablation/spmm", "filter3D"),
            ]

        elif plot_name == "rise_abl":
            INSETS = {
                0: ([10, 60], [0.25, 0.4, 0.6, 0.4], [3, 4]),
                1: ([10, 60], [0.25, 0.4, 0.6, 0.4], [3, 4]),
            }
            labels = [
                "BaCO",
                "No Hidden constraints",
                "No Feasibility limit",
            ]
            user_specified_methods = [
                "faes_hm",
                "base_hm",
                "faes0_hm",
            ]

            title = "TACO SpMM Abl"
            time_points = [20, 40, 60]

            tmp_data_sets: List[Tuple[str, str, List[str], int, int]] = [
                ("ablation/rise", "mm", user_specified_methods, 100, 11),
                ("ablation/rise", "scal", user_specified_methods, 60, 8),
            ]
            key_sets: List[Tuple[str, str]] = [
                ("ablation/rise", "mm"),
                ("ablation/rise", "scal"),
            ]

        data_sets = [
            {"Framework": tmp_d[0],
             "Operation": tmp_d[1],
             "Method": tmp_d[2],
             "Max_Evaluations": tmp_d[3],
             "Doe": tmp_d[4],
             } for tmp_d in tmp_data_sets
        ]

        data = get_data(data_sets, plot_name)

        if plot_type == "type1":
            make_geo(data, key_sets, time_points, plot_name)
        elif plot_type == "type2":
            make_geo2(data, key_sets, time_points, plot_name)
