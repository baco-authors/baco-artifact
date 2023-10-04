from __future__ import annotations
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import lines
import scipy
import math

pd.set_option("display.max_rows", None, "display.max_columns", None)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class Plot:

    """
    Reads and stores the data for the plots.
    """
    def __init__(self,
                 framework,
                 benchmark,
                 methods,
                 evals,
                 doe_samples,
                 shadows,
                 cutoff,
                 ):
        self.framework = framework
        self.benchmark = benchmark
        self.methods = methods
        self.evals = evals
        self.doe_samples = doe_samples
        self.shadows = shadows
        self.cutoff = cutoff
        self.times = [int(math.ceil((i + 1) * self.evals / 3)) for i in range(3)]
        self.data_folder = os.path.join(BASE_FOLDER, self.framework, self.benchmark)

        # Get the header names for different frameworks
        if self.framework in ["taco/spmm", "taco/sddmm", "taco/spmv", "taco/ttv", "taco/mttkrp"]:
            self.value_header = "compute_time"
            self.feasibility_header = None
            if self.framework in ["taco/mttkrp"]:
                self.expert = None
                self.default = None
            else:
                if self.framework in ["taco/ttv"]:
                    exp_col, def_col = 3, 2
                else:
                    exp_col, def_col = 2, 3
                try:
                    times = pd.read_csv(os.path.join(self.data_folder, "times.csv"))
                    self.expert = np.median(times.iloc[:, exp_col])
                    self.default = np.median(times.iloc[:, def_col])
                except:
                    self.expert = None
                    self.default = None
        elif self.framework == "rise":
            self.value_header = "runtime"
            self.feasibility_header = "Valid"
            try:
                with open(os.path.join(self.data_folder, "manual_configs", "default.csv")) as f:
                    self.default = float(f.readline())
                with open(os.path.join(self.data_folder, "manual_configs", "expert.csv")) as f:
                    self.expert = float(f.readline())
            except:
                self.expert = None
                self.default = None
        elif self.framework == "hpvm":
            self.value_header = "ExecTime"
            self.feasibility_header = "Valid"
            self.expert = None
            self.default = None
        else:
            raise Exception("incorrect framework")
        print(framework, benchmark)

        if framework == "taco/ttv" and benchmark in ["facebook", "random"]:
            self.feasibility_header = "Valid"
        if framework == "taco/ttv" and benchmark in ["uber3"]:
            self.expert = None
            self.default = None


        # Get the methods
        df = pd.DataFrame()
        val_df = pd.DataFrame()
        for method in self.methods:
            method_folder = os.path.join(self.data_folder, method)
            if not os.path.isdir(method_folder):
                continue
            if len([x for x in os.listdir(method_folder)]) == 1:
                method_folder = os.path.join(method_folder, os.listdir(method_folder)[0])

            files = [f for f in os.listdir(os.path.join(method_folder)) if ".csv" in f]
            if not files:
                continue
            mdf = pd.DataFrame()
            for i, f in enumerate(files):
                file_data = pd.read_csv(os.path.join(method_folder, f))
                tmp_feasibility_header = self.feasibility_header
                if tmp_feasibility_header is None:
                    file_data["Feasible"] = [True] * file_data.shape[0]
                    tmp_feasibility_header = "Feasible"
                file_data = file_data[[self.value_header, tmp_feasibility_header]]
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
                file_data = file_data.loc[file_data['Evaluations'] <= self.evals]
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

        ind = np.arange(df.shape[0])
        df.reindex(ind)
        if not val_df.empty:
            val_df = val_df[["Method", "Run", "Evaluations", "Min_Value"]]
            val_df.columns = ["Method", "Run", "Evaluations", "Value"]
            val_df.reindex(ind)
        self.stat_df = df
        self.value_df = val_df
        if self.framework == "hpvm" and not self.value_df.empty:
            self.value_df['Value'] = self.value_df['Value'].apply(lambda x: x * 1000)
            self.stat_df['Mean'] = self.stat_df['Mean'].apply(lambda x: x * 1000)
            self.stat_df['Std'] = self.stat_df['Std'].apply(lambda x: x * 1000)


def small_line_plots(
        plots
):
    """
    This is the plotting script. It takes a dict generated by get_data as input as well as a list[list[(FrameworkName, BenchmarkName)]], the outer list
    is for multiple figures and the inner list is for multiple plots within the same figure.
    """
    sns.set_theme()
    n_cols = 3
    n_rows = int(len(plots) / 3)

    plt.rcParams["figure.figsize"] = [12, 9]

    tickfontsize = 12
    suptitlefontsize = 20
    titlefontsize2 = 14
    axislabelfontsize = 16
    legendfontsize = 14
    legendanchor = (0.5, 0.95)
    labelpad = 0


    bot_limit = 0.11
    top_limit = 0.88
    total_fig_height = top_limit - bot_limit
    bot_height = 0.55 * total_fig_height / n_rows
    break_height = 0.01 * total_fig_height / n_rows
    top_height = 0.2 * total_fig_height / n_rows
    figure_bots = [bot_limit + i * total_fig_height / n_rows for i in range(n_rows)]
    figure_break_bots = [figure_bots[i] + bot_height for i in range(n_rows)]
    figure_break_tops = [figure_break_bots[i] + break_height for i in range(n_rows)]

    left = 0.09
    hl = 0.25
    hs = 0.04

    axs = {}
    fig = plt.figure()
    for j in range(n_cols):
        for i in range(n_rows):
            axs[(i, j, 1)] = fig.add_axes([left + j * (hl + hs), figure_break_tops[i], hl, top_height])
            axs[(i, j, 0)] = fig.add_axes([left + j * (hl + hs), figure_bots[i], hl, bot_height])

    label_dict = {l: None for l in labels + ["Expert Configuration", "Default Configuration", "DoE"]}

    # create the individual plots
    experts = []
    for plot_index, p in enumerate(plots):
        df = p.stat_df
        if df.empty:
            continue
        methods = p.methods
        doe = p.doe_samples
        axl = axs[(n_rows - 1 - int(math.floor(plot_index / n_cols)), plot_index % n_cols, 0)]
        axu = axs[(n_rows - 1 - int(math.floor(plot_index / n_cols)), plot_index % n_cols, 1)]
        min_regret = np.inf
        max_regret = -np.inf
        n_Evaluations = 0
        palette = sns.color_palette()

        if not p.expert is None:
            label_dict["Expert Configuration"] = axl.axhline(y=p.expert, color="grey", linewidth=1.5)
            min_regret = min(min_regret, p.expert)

        default = p.default

        for im, method in enumerate(methods):
            if not "Method" in df.columns:
                continue
            df_method = df.loc[df.Method == method]
            label_dict[labels[im]] = axu.plot(df_method.Evaluations, df_method.Mean, label=method, color=palette[im])[0]
            axl.plot(df_method.Evaluations, df_method.Mean, label=method, color=palette[im], linewidth=2)
            if p.shadows:
                axu.fill_between(df_method.Evaluations, df_method.Mean - df_method.Std, df_method.Mean + df_method.Std, alpha=0.2, label='_nolegend_', color=palette[im])
                axl.fill_between(df_method.Evaluations, df_method.Mean - df_method.Std, df_method.Mean + df_method.Std, alpha=0.2, label='_nolegend_', color=palette[im])
            min_regret = min(min_regret, np.min(df_method.Mean))
            max_regret = max(max_regret, np.max(df_method.Mean))
            n_Evaluations = max(n_Evaluations, np.max(df_method.Evaluations))
            if default is None:
                default = df_method.Mean[0]
            if not p.expert is None:
                mean_vector = df_method.Mean.to_numpy()
                first_meet = [i for i in range(len(mean_vector) - 1) if mean_vector[i] <= p.expert]
                if first_meet:
                    axl.scatter([first_meet[1]], [p.expert], marker='*', s=200, facecolor=palette[im], edgecolor="black", label='_nolegend_', zorder=5)

        #
        label_dict["Default Configuration"] = axu.axhline(y=default, color="grey", linewidth=1.5, linestyle="dashdot", )
        max_regret = max(max_regret, default)

        axl.spines["top"].set_color("grey")
        axl.spines["top"].set_linewidth(.5)
        axu.spines["bottom"].set_color("grey")
        axu.spines["bottom"].set_linewidth(.5)
        print(p.benchmark, p.expert)
        if not p.expert == None:
            break_point = p.cutoff * p.expert
            u_interval = max_regret - break_point
            l_interval = break_point - p.expert
            ul = (break_point, max_regret + 0.15 * u_interval)
            ll = (min(p.expert - 0.2 * l_interval, 0.95 * min_regret), break_point)
        else:
            break_point = p.cutoff * min_regret
            u_interval = max_regret - break_point
            l_interval = break_point - min_regret
            ul = (break_point, max_regret + 0.15 * u_interval)
            ll = (min(min_regret - 0.2 * l_interval, 0.95 * min_regret), break_point)

        line = lines.Line2D((doe, doe), (ll[0], ll[1] + 0.4 * (ll[1] - ll[0])), color="grey", linewidth=1.5, linestyle="dotted", clip_on=False)
        label_dict["DoE"] = axl.add_line(line)
        axu.set_ylim(*ul)
        axl.set_ylim(*ll)
        axu.set_title(p.benchmark, fontsize=titlefontsize2)
        axu.set(xlabel=None)
        axu.tick_params(bottom=False, labelbottom=False)
        axl.tick_params(axis='x', labelsize=tickfontsize, pad=labelpad)
        axl.tick_params(axis='y', labelsize=tickfontsize, pad=labelpad)
        axu.tick_params(axis='y', labelsize=tickfontsize, pad=labelpad)

    fig.text(0.5, 0.03, XLABEL, ha='center', fontsize=axislabelfontsize)
    fig.text(0.02, 0.5, YLABEL, va='center', rotation='vertical', fontsize=axislabelfontsize)

    title = "Evaluation of Average Best Runtime"
    plt.suptitle(title + " (Lower is better)", fontsize=suptitlefontsize)
    leg = plt.legend(
        handles=label_dict.values(),
        labels=label_dict.keys(),
        loc="upper center",
        bbox_to_anchor=legendanchor,
        fancybox=True,
        shadow=True,
        ncol=4,
        bbox_transform=fig.transFigure,
        fontsize=legendfontsize,
        facecolor="white",
        handlelength=2.7,
    )
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)

    fig_name = f"plots/line_plot_small.pdf"
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')


def large_line_plots(
        plots
):
    """
    This is the plotting script. It takes a dict generated by get_data as input as well as a list[list[(FrameworkName, BenchmarkName)]], the outer list
    is for multiple figures and the inner list is for multiple plots within the same figure.
    """
    sns.set_theme()
    n_cols = 3
    n_rows = int(len(plots) / 3)

    plt.rcParams["figure.figsize"] = [12, 16]
    tickfontsize = 12
    suptitlefontsize = 20
    titlefontsize2 = 14
    axislabelfontsize = 16
    legendfontsize = 14
    legendanchor = (0.5, 0.95)
    labelpad = 0


    bot_limit = 0.11
    top_limit = 0.88
    total_fig_height = top_limit - bot_limit
    bot_height = 0.55 * total_fig_height / n_rows
    break_height = 0.01 * total_fig_height / n_rows
    top_height = 0.2 * total_fig_height / n_rows
    figure_bots = [bot_limit + i * total_fig_height / n_rows for i in range(n_rows)]
    figure_break_bots = [figure_bots[i] + bot_height for i in range(n_rows)]
    figure_break_tops = [figure_break_bots[i] + break_height for i in range(n_rows)]

    left = 0.09
    hl = 0.25
    hs = 0.04

    axs = {}
    fig = plt.figure()
    for j in range(n_cols):
        for i in range(n_rows):
            axs[(i, j, 1)] = fig.add_axes([left + j * (hl + hs), figure_break_tops[i], hl, top_height])
            axs[(i, j, 0)] = fig.add_axes([left + j * (hl + hs), figure_bots[i], hl, bot_height])

    label_dict = {l: None for l in labels + ["Expert Configuration", "Default Configuration", "DoE"]}

    # create the individual plots
    for plot_index, p in enumerate(plots):
        df = p.stat_df
        if df.empty:
            continue
        methods = p.methods
        doe = p.doe_samples
        axl = axs[(n_rows - 1 - int(math.floor(plot_index / n_cols)), plot_index % n_cols, 0)]
        axu = axs[(n_rows - 1 - int(math.floor(plot_index / n_cols)), plot_index % n_cols, 1)]
        min_regret = np.inf
        max_regret = -np.inf
        n_Evaluations = 0
        palette = sns.color_palette()

        if not p.expert is None:
            label_dict["Expert Configuration"] = axl.axhline(y=p.expert, color="grey", linewidth=1.5)  # , linestyle="dotted",)
            min_regret = min(min_regret, p.expert)

        default = p.default

        for im, method in enumerate(methods):
            if not "Method" in df.columns:
                continue
            df_method = df.loc[df.Method == method]
            label_dict[labels[im]] = axu.plot(df_method.Evaluations, df_method.Mean, label=method, color=palette[im])[0]
            axl.plot(df_method.Evaluations, df_method.Mean, label=method, color=palette[im], linewidth=2)
            if p.shadows:
                axu.fill_between(df_method.Evaluations, df_method.Mean - df_method.Std, df_method.Mean + df_method.Std, alpha=0.2, label='_nolegend_', color=palette[im])
                axl.fill_between(df_method.Evaluations, df_method.Mean - df_method.Std, df_method.Mean + df_method.Std, alpha=0.2, label='_nolegend_', color=palette[im])
            min_regret = min(min_regret, np.min(df_method.Mean))
            max_regret = max(max_regret, np.max(df_method.Mean))
            n_Evaluations = max(n_Evaluations, np.max(df_method.Evaluations))
            if default is None:
                default = df_method.Mean[0]
            if not p.expert is None:
                mean_vector = df_method.Mean.to_numpy()
                first_meet = [i for i in range(len(mean_vector) - 1) if mean_vector[i] <= p.expert]
                if first_meet:
                    axl.scatter([first_meet[1]], [p.expert], marker='*', s=200, facecolor=palette[im], edgecolor="black", label='_nolegend_', zorder=5)

        #
        label_dict["Default Configuration"] = axu.axhline(y=default, color="grey", linewidth=1.5, linestyle="dashdot", )
        max_regret = max(max_regret, default)

        axl.spines["top"].set_color("grey")
        axl.spines["top"].set_linewidth(.5)
        axu.spines["bottom"].set_color("grey")
        axu.spines["bottom"].set_linewidth(.5)

        if not p.expert == None:
            break_point = p.cutoff * p.expert
            u_interval = max_regret - break_point
            l_interval = break_point - p.expert
            ul = (break_point, max_regret + 0.15 * u_interval)
            ll = (min(p.expert - 0.2 * l_interval, 0.95 * min_regret), break_point)
        else:
            break_point = p.cutoff * min_regret
            u_interval = max_regret - break_point
            l_interval = break_point - min_regret
            ul = (break_point, max_regret + 0.15 * u_interval)
            ll = (min(min_regret - 0.2 * l_interval, 0.95 * min_regret), break_point)

        line = lines.Line2D((doe, doe), (ll[0], ll[1] + 0.4 * (ll[1] - ll[0])), color="grey", linewidth=1.5, linestyle="dotted", clip_on=False)
        label_dict["DoE"] = axl.add_line(line)
        axu.set_ylim(*ul)
        axl.set_ylim(*ll)
        axu.set_title(p.benchmark, fontsize=titlefontsize2)
        axu.set(xlabel=None)
        axu.tick_params(bottom=False, labelbottom=False)
        axl.tick_params(axis='x', labelsize=tickfontsize, pad=labelpad)
        axl.tick_params(axis='y', labelsize=tickfontsize, pad=labelpad)
        axu.tick_params(axis='y', labelsize=tickfontsize, pad=labelpad)

    fig.text(0.5, 0.03, XLABEL, ha='center', fontsize=axislabelfontsize)
    fig.text(0.02, 0.5, YLABEL, va='center', rotation='vertical', fontsize=axislabelfontsize)

    title = "Evaluation of Average Best Runtime"
    plt.suptitle(title + " (Lower is better)", fontsize=suptitlefontsize)
    leg = plt.legend(
        handles=label_dict.values(),
        labels=label_dict.keys(),
        loc="upper center",
        bbox_to_anchor=legendanchor,
        fancybox=True,
        shadow=True,
        ncol=4,
        bbox_transform=fig.transFigure,
        fontsize=legendfontsize,
        facecolor="white",
        handlelength=2.7,
    )
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)

    fig_name = f"plots/line_plot_large.pdf"
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')


def bar_plot(plots):

    plt.rcParams["figure.figsize"] = [6, 9]
    sns.set_theme()

    fig = plt.figure()
    titlefontsize = 17
    axislabelfontsize = 16
    legendanchor = (0.5, 1.9)
    n_rows = 3

    titles = ["HPVM2FPGA", "RISE \& ELEVATE", "TACO"]
    num_titles = 3

    bot_limit = 0.09
    top_limit = 0.88
    title_space = 0.02
    title_space2 = 0.02
    total_fig_height = top_limit - bot_limit - title_space * num_titles
    fig_height = 0.72 * total_fig_height / n_rows
    figure_bots = [bot_limit + i * total_fig_height / n_rows for i in range(n_rows)]
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            if titles[i] != None:
                figure_bots[j] += title_space
    figure_tops = [figure_bots[i] + fig_height for i in range(n_rows)]

    left = 0.09
    width = 0.85

    axs = []
    for i in range(n_rows):
        axs.append(fig.add_axes([left, figure_bots[i], width, fig_height]))

    for i, plot_group in enumerate((
            [plots[22], plots[23], plots[24]],
            [plots[x] for x in range(15, 22)],
            [plots[x] for x in range(15)]

    )):

        merged_df = pd.DataFrame()
        geo_df = pd.DataFrame()
        for plot in plot_group:
            stat_df = plot.stat_df
            if stat_df.empty:
                continue

            if plot.expert == None:
                plot.expert = stat_df.loc[stat_df["Evaluations"] == plot.evals, "Mean"].values[0]

            if plot.default is None:
                plot.default = stat_df.loc[stat_df["Evaluations"] == 1, "Mean"].values[0]

            for tidx, t in enumerate(plot.times):
                df = stat_df.loc[stat_df["Evaluations"] == t]
                df["imp_exp"] = df.apply(lambda row: plot.expert / row["Mean"], axis=1)
                df["imp_def"] = df.apply(lambda row: plot.default / row["Mean"] + 0.01, axis=1)
                df["benchmark"] = [plot.benchmark] * df.shape[0]
                df["Evaluations"] = [f"{math.floor(100 * (1 + tidx) / 3)}\%"] * df.shape[0]
                merged_df = merged_df.append(df)

                row = {"Evaluations": f"{math.floor(100 * (1 + tidx) / 3)}\%",
                       "benchmark": plot.benchmark,
                       "Method": "Default",
                       "imp_exp": plot.expert / plot.default + 0.01,
                       "imp_def": 1,
                       }
                merged_df = merged_df.append(row, ignore_index=True)

        if merged_df.empty:
            continue

        for a, b in zip(plot.methods, labels):
            merged_df.replace(to_replace=a, value=b, inplace=True)

        for m in merged_df["Method"].unique():
            for t in ["33\%", "66\%", "100\%"]:
                t_df = merged_df.loc[merged_df["Method"] == m]
                t_df = t_df.loc[merged_df["Evaluations"] == t]
                row_geo = {"Method": m,
                           "imp_exp": scipy.stats.gmean(t_df["imp_exp"]),
                           "imp_def": scipy.stats.gmean(t_df["imp_def"]),
                           "Evaluations": t
                           }
                geo_df = geo_df.append(row_geo, ignore_index=True)

        print(geo_df.columns)
        ax = axs[i]
        geo_df.columns = ["Method", "Speedup over Expert", "Speedup over Default", "Evaluations"]
        for a, b in zip(["33\%", "66\%", "100\%"], ["tiny", "small", "full"]):
            geo_df.replace(to_replace=a, value=b, inplace=True)
        sns.barplot(data=geo_df, x="Evaluations", hue="Method", y="Speedup over Expert", ax=ax, ci="sd", errwidth=.5, errcolor="grey")
        ax.axhline(1, linestyle="dashed", color="grey", linewidth=.8)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{'{:.1f}'.format(tick)}x" for tick in yticks])
        ax.tick_params(axis='y', labelsize=12, pad=0.01)
        ax.tick_params(axis='x', labelsize=14, pad=0.01)
        ax.set(xlabel=None)
        ax.set(ylabel=None)

        fig.text(0.5, 0.03, "Autotuning Budget", ha='center', fontsize=axislabelfontsize)

        if i == 2:
            ax.legend(
                title="",
                loc="upper center",
                bbox_to_anchor=legendanchor,
                ncol=2,
                fontsize=14,
                fancybox=True,
                shadow=True,
                facecolor="white"
            )

    fig.suptitle("Average Performance relative to Expert", fontsize=17)
    for i, t in enumerate(titles):
        if t != None:
            fig.text(0.5, figure_tops[i] + title_space2, r"%s" % t, ha='center', fontsize=titlefontsize)

    try:
        axs[0].get_legend().remove()
        axs[1].get_legend().remove()
    except:
        pass
    plt.savefig(f"plots/bar.pdf", dpi=600)



if __name__ == "__main__":


    BASE_FOLDER = "results"
    YLABEL = "Runtime [ms]"
    XLABEL = "Number of Configuration Evaluations"
    plots = []

    labels = ["BaCO", "ATF with OpenTuner", "Ytopt", "Uniform Sampling", "CoT Sampling"]

    # TACO
    methods = ["bayesian_optimization", "opentuner", "ytopt", "random_sampling", "embedding_random_sampling"]

    # spmm
    for matrix in ["scircuit", "cage12", "laminar\_duct3D"]:
        plots.append(Plot("taco/spmm", matrix, methods, 60, 7, shadows=False, cutoff=5))

    # sddmm
    for matrix in ["email-Enron", "ACTIVSg10K", "Goodwin\_040"]:
        plots.append(Plot("taco/sddmm", matrix, methods, 60, 7, shadows=False, cutoff=1.7))

    # mttkrp
    for matrix in ["uber", "nips", "chicago"]:
        plots.append(Plot("taco/mttkrp", matrix, methods, 60, 7, shadows=False, cutoff=5))

    # TTV
    for matrix in ["facebook", "uber3", "random"]:
        plots.append(Plot("taco/ttv", matrix, methods, 60, 8, shadows=False, cutoff=5))

    # spmv
    matrices = ["laminar\_duct3D", "cage12", "filter3D"]
    for matrix in matrices:
        plots.append(Plot("taco/spmv", matrix, methods, 60, 7, shadows=False, cutoff=3))

    tmp_plot = Plot("taco/spmv", "random", methods, 60, 7, shadows=False, cutoff=3)

    # rise
    methods = ["bolog_cot", "opentuner", "ytoptccs", "rs_cot", "rs_emb"]

    plots.append(Plot("rise", "MM_CPU", methods, 100, 6, shadows=False, cutoff=2))
    plots.append(Plot("rise", "MM_GPU", methods, 120, 11, shadows=False, cutoff=10))
    plots.append(Plot("rise", "Asum_GPU", methods, 60, 6, shadows=False, cutoff=10))
    plots.append(Plot("rise", "Scal_GPU", methods, 60, 8, shadows=False, cutoff=20))
    plots.append(Plot("rise", "K-means_GPU", methods, 60, 5, shadows=False, cutoff=5))
    plots.append(Plot("rise", "Harris_GPU", methods, 100, 9, shadows=False, cutoff=5))
    plots.append(Plot("rise", "Stencil_GPU", methods, 60, 6, shadows=False, cutoff=2))

    # hpvm
    methods = ["out_bo", "out_ot", "out_ytoptccs", "out_rs", "out_ers"]

    plots.append(Plot("hpvm", "BFS", methods, 20, 5, shadows=False, cutoff=4))
    plots.append(Plot("hpvm", "Audio", methods, 60, 16, shadows=False, cutoff=1.5))
    plots.append(Plot("hpvm", "PreEuler", methods, 60, 8, shadows=False, cutoff=3))

    plots.append(tmp_plot)

    for x in [0, 1, 2]:
        plots[x].benchmark = r" TACO SpMM " + plots[x].benchmark

    for x in [3, 4, 5]:
        plots[x].benchmark = r" TACO SDDMM " + plots[x].benchmark

    for x in [6, 7, 8]:
        plots[x].benchmark = r" TACO MTTKRP " + plots[x].benchmark

    for x in [9, 10, 11]:
        plots[x].benchmark = r" TACO TTV " + plots[x].benchmark

    for x in [12, 13, 14, 25]:
        plots[x].benchmark = r" TACO SpMV " + plots[x].benchmark

    for x in [15, 16, 17, 18, 19, 20, 21]:
        plots[x].benchmark = r" RISE/ELEVATE " + plots[x].benchmark

    for x in [22, 23, 24]:
        plots[x].benchmark = r" HPVM2FPGA " + plots[x].benchmark


    # SMALL LINE PLOT
    small_plots = [plots[x] for x in [0, 9, 13, 15, 16, 17, 22, 23, 24]]
    small_line_plots(small_plots)


    # LARGE LINE PLOT
    large_plots = [plots[x] for x in [3, 4, 5, 6, 7, 8, 1, 2, 10, 12, 25, 18, 19, 20, 21]]
    large_line_plots(large_plots)

    # BAR PLOT
    bar_plot(plots)

