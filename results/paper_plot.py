import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from scipy.stats import wasserstein_distance

#
#
# def UNSW_data_distribution_plots():
#     # Class names and sizes
#     class_names = [
#         "Analysis", "Backdoor", "Benign", "DoS", "Exploits",
#         "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"
#     ]
#     # Format long names to fit in two lines
#     formatted_names = [name if len(name) <= 10 else name[:6] + '\n' + name[6:] for name in class_names]
#
#     training_sizes = [2142, 1863, 1775011, 13082, 35360, 19397, 172385, 11189, 1209, 139]
#     testing_sizes = [535, 466, 443753, 3271, 8905, 4849, 43096, 2798, 302, 35]
#
#     x = np.arange(len(class_names))
#     width = 0.35  # Bar width
#
#     font = {'family': 'Arial',
#             'weight': 'normal',
#             'size': 25}
#     matplotlib.rc('font', **font)
#     plt.rc('font', **font)
#
#     # Create broken-axis subplots
#     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 10), gridspec_kw={'height_ratios': [2, 3]})
#
#     # Top plot: zoom in on large values
#     ax1.bar(x - width/2, training_sizes, width, label='Train', color='orange')
#     ax1.bar(x + width/2, testing_sizes, width, label='Test', color='orangered')
#     ax1.set_ylim(400000, 1800000)
#     ax1.set_yticks([500000, 1000000, 1500000])
#     ax1.ticklabel_format(style='plain', axis='y')
#
#     # Bottom plot: zoom in on small values
#     ax2.bar(x - width/2, training_sizes, width, color='orange')
#     ax2.bar(x + width/2, testing_sizes, width, color='orangered')
#     ax2.set_ylim(0, 50000)
#     ax2.set_yticks([0, 10000, 20000, 30000, 40000])
#     ax2.ticklabel_format(style='plain', axis='y')
#
#     # X-axis formatting
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(formatted_names, rotation=0, ha='center')
#
#     # Hide spines for broken axis effect
#     ax1.spines['bottom'].set_visible(False)
#     ax2.spines['top'].set_visible(False)
#     ax1.tick_params(labeltop=False)
#     ax2.tick_params(labeltop=False)
#
#     # Draw slashes to indicate broken axis
#     d = .5
#     kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
#                   linestyle='none', color='k', mec='k', mew=1, clip_on=False)
#     ax1.plot(np.arange(-0.5, len(class_names)), np.full(len(class_names)+1, ax1.get_ylim()[0]), **kwargs)
#     ax2.plot(np.arange(-0.5, len(class_names)), np.full(len(class_names)+1, ax2.get_ylim()[1]), **kwargs)
#
#     # Title and unified legend
#     # fig.suptitle("UNSW-NB15 Data Distribution (Training vs Testing)")
#     handles, labels = ax1.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper right', fontsize=35, framealpha=0.3)
#
#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.1)
#     plt.savefig("plots/UNSW_data_distribution.pdf", bbox_inches='tight')
#     plt.show()


def UNSW_data_distribution_plots_V2():
    # Class names and sizes
    class_names = [
        "Analysis", "Backdoor", "Benign", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"
    ]
    formatted_names = [name if len(name) <= 7 else name[:6] + '\n' + name[6:] for name in class_names]

    training_sizes = [2142, 1863, 1775011, 13082, 35360, 19397, 172385, 11189, 1209, 139]
    testing_sizes  = [535, 466, 443753, 3271, 8905, 4849, 43096, 2798, 302, 35]

    x = np.arange(len(class_names))
    width = 0.35

    # Font settings
    font = {'family': 'Arial', 'weight': 'normal', 'size': 35}
    matplotlib.rc('font', **font)
    plt.rc('font', **font)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8), gridspec_kw={'height_ratios': [2, 3]})

    # Top plot for large values
    ax1.bar(x - width/2, training_sizes, width, label='Train', color='orange')
    ax1.bar(x + width/2, testing_sizes, width, label='Test', color='orangered')
    ax1.set_ylim(400000, 2000000)
    ax1.set_yticks([500000, 1000000, 1500000])
    ax1.ticklabel_format(style='plain', axis='y')

    # Bottom plot for small values
    ax2.bar(x - width/2, training_sizes, width, color='orange')
    ax2.bar(x + width/2, testing_sizes, width, color='orangered')
    ax2.set_ylim(0, 50000)
    ax2.set_yticks([0, 10000, 20000, 30000, 40000])
    ax2.ticklabel_format(style='plain', axis='y')

    # X-axis formatting
    ax2.set_xticks(x)
    ax2.set_xticklabels(formatted_names, rotation=90, ha='center')

    # Broken axis formatting
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)
    ax2.tick_params(labeltop=False)

    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle='none', color='k', mec='k', mew=1, clip_on=False)
    ax1.plot(np.arange(-0.5, len(class_names)), np.full(len(class_names)+1, ax1.get_ylim()[0]), **kwargs)
    ax2.plot(np.arange(-0.5, len(class_names)), np.full(len(class_names)+1, ax2.get_ylim()[1]), **kwargs)

    ax1.margins(x=0)
    ax2.margins(x=0)
    # ax2.set_xticklabels(formatted_names, rotation=45, ha='right')

    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=35, framealpha=0.3, bbox_to_anchor=(0.97, 0.95))  # adjust x (left–right), y (bottom–top)

    # Save and show
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig("plots/UNSW_data_distribution.pdf", bbox_inches='tight')
    plt.show()




def CICIDS_data_distribution_plots():
    # Class names and sizes
    class_names = [
        "BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye",
        "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttp", "Bot"
    ]
    # Format long names into two lines
    formatted_names = [name if len(name) <= 7 else name[:6] + '\n' + name[6:] for name in class_names]

    training_sizes = [1817232, 183889, 126994, 102390, 8253, 6362, 4766, 4604, 4397, 1595]
    testing_sizes  = [454088, 46235, 31810, 25635, 2040, 1573, 1131, 1192, 1102, 361]

    x = np.arange(len(class_names))
    width = 0.35  # Bar width

    # Set global font
    font = {'family': 'Arial', 'weight': 'normal', 'size': 35}
    matplotlib.rc('font', **font)
    plt.rc('font', **font)

    # Create broken-axis bar plots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8), gridspec_kw={'height_ratios': [2, 3]})

    # Top (high values)
    ax1.bar(x - width/2, training_sizes, width, label='Train', color='orange')
    ax1.bar(x + width/2, testing_sizes, width, label='Test', color='orangered')
    ax1.set_ylim(400000, 2000000)
    ax1.set_yticks([500000, 1000000, 1500000])
    ax1.ticklabel_format(style='plain', axis='y')

    # Bottom (low values)
    ax2.bar(x - width/2, training_sizes, width, color='orange')
    ax2.bar(x + width/2, testing_sizes, width, color='orangered')
    ax2.set_ylim(0, 50000)
    ax2.set_yticks([0, 10000, 20000, 30000, 40000])
    ax2.ticklabel_format(style='plain', axis='y')

    # X-axis formatting
    ax2.set_xticks(x)
    ax2.set_xticklabels(formatted_names, rotation=90, ha='center')

    # Broken axis styling
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)
    ax2.tick_params(labeltop=False)

    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle='none', color='k', mec='k', mew=1, clip_on=False)
    ax1.plot(np.arange(-0.5, len(class_names)), np.full(len(class_names)+1, ax1.get_ylim()[0]), **kwargs)
    ax2.plot(np.arange(-0.5, len(class_names)), np.full(len(class_names)+1, ax2.get_ylim()[1]), **kwargs)

    # Add single legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=35, framealpha=0.3, bbox_to_anchor=(0.97, 0.95))

    ax1.get_yaxis().set_visible(True)
    ax2.get_yaxis().set_visible(True)
    ax1.margins(x=0)
    ax2.margins(x=0)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    # Save as PDF
    plt.savefig("plots/CICIDS_data_distribution.pdf", bbox_inches='tight')
    plt.show()



def CICDDoS2019_data_distribution_plots():
    class_names = [
        "BENIGN", "UDP-lag", "DrDoS_SSDP", "DrDoS_DNS", "DrDoS_MSSQL",
        "DrDoS_NetBIOS", "DrDoS_LDAP", "DrDoS_NTP", "DrDoS_UDP", "Syn", "DrDoS_SNMP"
    ]
    formatted_names = [name if len(name) <= 7 else name[:6] + '\n' + name[6:] for name in class_names]

    training_sizes = [800000] + [80000] * 10
    testing_sizes = [200000] + [20000] * 10

    x = np.arange(len(class_names))
    width = 0.35

    font = {'family': 'Arial', 'weight': 'normal', 'size': 35}
    matplotlib.rc('font', **font)
    plt.rc('font', **font)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8), gridspec_kw={'height_ratios': [2, 3]})

    ax1.bar(x - width/2, training_sizes, width, label='Train', color='orange')
    ax1.bar(x + width/2, testing_sizes, width, label='Test', color='orangered')
    ax1.set_ylim(400000, 2000000)
    ax1.set_yticks([500000, 1000000, 1500000])
    ax1.ticklabel_format(style='plain', axis='y')

    ax2.bar(x - width/2, training_sizes, width, color='orange')
    ax2.bar(x + width/2, testing_sizes, width, color='orangered')
    ax2.set_ylim(0, 50000)
    ax2.set_yticks([0, 10000, 20000, 30000, 40000])
    ax2.ticklabel_format(style='plain', axis='y')

    ax2.set_xticks(x)
    ax2.set_xticklabels(formatted_names, rotation=90, ha='center')

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)
    ax2.tick_params(labeltop=False)

    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle='none', color='k', mec='k', mew=1, clip_on=False)
    ax1.plot(np.arange(-0.5, len(class_names)), np.full(len(class_names)+1, ax1.get_ylim()[0]), **kwargs)
    ax2.plot(np.arange(-0.5, len(class_names)), np.full(len(class_names)+1, ax2.get_ylim()[1]), **kwargs)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=35, framealpha=0.3, bbox_to_anchor=(0.97, 0.95))

    ax1.margins(x=0)
    ax2.margins(x=0)
    ax1.get_yaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig("plots/CICDDoS2019_data_distribution.pdf", bbox_inches='tight')
    plt.show()


def combined_distribution_plots_tight():
    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 30}
    matplotlib.rc('font', **font)

    fig, axes = plt.subplots(2, 2, sharey='row', figsize=(28, 6),
                             gridspec_kw={'height_ratios': [2, 3], 'wspace': 0.01, 'hspace': 0.05})

    # CICIDS2017
    class_names1 = [
        "BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye",
        "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttp", "Bot"
    ]
    formatted_names1 = [name if len(name) <= 7 else name[:6] + '\n' + name[6:] for name in class_names1]
    train1 = [1817232, 183889, 126994, 102390, 8253, 6362, 4766, 4604, 4397, 1595]
    test1 =  [454088, 46235, 31810, 25635, 2040, 1573, 1131, 1192, 1102, 361]
    x1 = np.arange(len(class_names1))
    width = 0.35

    ax1, ax2 = axes[0, 0], axes[1, 0]
    ax1.bar(x1 - width/2, train1, width, color='#619CFF', label='Train')
    ax1.bar(x1 + width/2, test1, width, color='#00A86B', label='Test')
    ax1.set_ylim(400000, 2000000)
    ax1.set_yticks([500000, 1000000, 1500000])
    ax1.ticklabel_format(style='plain', axis='y')
    ax2.bar(x1 - width/2, train1, width, color='#619CFF')
    ax2.bar(x1 + width/2, test1, width, color='#00A86B')
    ax2.set_ylim(0, 50000)
    ax2.set_xticks(x1)
    ax2.set_xticklabels(formatted_names1, rotation=90, ha='center')
    ax2.set_yticks([0, 10000, 20000, 30000, 40000])
    ax2.ticklabel_format(style='plain', axis='y')
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)
    ax2.tick_params(labeltop=False)
    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle='none', color='k', mec='k', mew=1, clip_on=False)
    ax1.plot(np.arange(-0.5, len(class_names1)), np.full(len(class_names1)+1, ax1.get_ylim()[0]), **kwargs)
    ax2.plot(np.arange(-0.5, len(class_names1)), np.full(len(class_names1)+1, ax2.get_ylim()[1]), **kwargs)
    ax2.set_ylabel("Number of Flows", labelpad=20)

    # CICDDoS2019
    class_names2 = [
        "BENIGN", "UDP-lag", "DrDoS_SSDP", "DrDoS_DNS", "DrDoS_MSSQL",
        "DrDoS_NetBIOS", "DrDoS_LDAP", "DrDoS_NTP", "DrDoS_UDP", "Syn", "DrDoS_SNMP"
    ]
    formatted_names2 = [name if len(name) <= 7 else name[:6] + '\n' + name[6:] for name in class_names2]
    train2 = [800000] + [80000] * 10
    test2 = [200000] + [20000] * 10
    x2 = np.arange(len(class_names2))

    ax3, ax4 = axes[0, 1], axes[1, 1]
    ax3.bar(x2 - width/2, train2, width, color='#619CFF')
    ax3.bar(x2 + width/2, test2, width, color='#00A86B')
    ax3.set_ylim(400000, 2000000)
    ax3.set_yticks([500000, 1000000, 1500000])
    ax3.set_yticklabels(["500k", "1,000k", "1,500k"])
    # ax3.ticklabel_format(style='plain', axis='y')
    ax4.bar(x2 - width/2, train2, width, color='#619CFF')
    ax4.bar(x2 + width/2, test2, width, color='#00A86B')
    ax4.set_ylim(0, 50000)
    ax4.set_xticks(x2)
    ax4.set_xticklabels(formatted_names2, rotation=90, ha='center')
    ax4.set_yticks([0, 10000, 20000, 30000, 40000])
    ax4.set_yticklabels([0, "10k", "20k", "30k", "40k"])
    # ax4.ticklabel_format(style='plain', axis='y')
    ax3.spines['bottom'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax3.tick_params(labeltop=False)
    ax4.tick_params(labeltop=False)
    ax3.plot(np.arange(-0.5, len(class_names2)), np.full(len(class_names2)+1, ax3.get_ylim()[0]), **kwargs)
    ax4.plot(np.arange(-0.5, len(class_names2)), np.full(len(class_names2)+1, ax4.get_ylim()[1]), **kwargs)

    ax1.set_xticklabels([])
    ax3.set_xticklabels([])
    ax1.set_title("CICIDS2017")
    ax3.set_title("CICDDoS2019")
    # Remove y-axis on right-side plots to save horizontal space
    ax3.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.51, 0.7), fontsize=30, framealpha=0.4, ncol=2)

    plt.subplots_adjust(hspace=0.05, wspace=0.01)
    matplotlib.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.savefig("plots/combined_distribution_plots_tight.pdf", bbox_inches='tight')
    plt.show()





def plot_pca_density_from_dataframe(datasets: dict, k_components=1, save_path=None):
    """
    Plot PCA density distributions for multiple datasets on the same figure.

    Parameters:
    - datasets: dict of {label: np.ndarray or pd.DataFrame}
    - k_components: number of PCA components (must be 1 for density plot)
    - save_path: optional string to save the figure as a PDF
    """
    assert k_components == 1, "Only k_components=1 is supported for density plot"

    # Convert to NumPy and ensure all values are aligned
    arrays = {label: np.asarray(data) for label, data in datasets.items()}
    all_data = np.vstack(list(arrays.values()))

    # Fit PCA on all data
    pca = PCA(n_components=k_components)
    pca.fit(all_data)

    # Project data
    projections = {label: pca.transform(arr)[:, 0] for label, arr in arrays.items()}

    # Determine common x-axis range
    all_proj_flat = np.concatenate(list(projections.values()))
    # x_center = (all_proj_flat.max() + all_proj_flat.min()) / 2
    # x_range = (all_proj_flat.max() - all_proj_flat.min()) / 2
    # x_min = x_center - 1.2 * x_range
    # x_max = x_center + 1.2 * x_range
    # Add a small margin instead of using fixed 1.2x range
    margin = 0.01 * (all_proj_flat.max() - all_proj_flat.min())
    x_min = all_proj_flat.min() - margin
    x_max = all_proj_flat.max() + margin

    # Start plotting
    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 35}
    matplotlib.rc('font', **font)
    plt.rc('font', **font)

    plt.figure(figsize=(8, 6))
    # for label, proj in projections.items():
    #     sns.kdeplot(proj, label=label, fill=True, linewidth=1)
    for label, proj in projections.items():
        sns.kdeplot(proj, label=label, fill=True, linewidth=1, cut=0.1)

    plt.xlabel('PCA Projection')
    # plt.ylabel('')
    plt.ylabel('Density')
    plt.xlim(x_min, x_max)
    # plt.xticks([0, 2, 4])
    plt.xticks([-1, 0, 1])
    # plt.yticks([0, 0.5, 1, 1.5])
    plt.yticks([0, 1, 2, 3])
    plt.legend(fontsize=20, loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Saved plot to {save_path}")
    plt.show()


def plot_two_pca_density_figures_side_by_side(dataset_list1: dict, dataset_list2: dict, k_components=1, save_path=None):
    """
    Plot two PCA density plots side by side in separate subplots without merging data.

    Each subplot handles its own PCA and KDE plot independently.

    Parameters:
    - dataset_list1: dict of {label: np.ndarray} for left subplot
    - dataset_list2: dict of {label: np.ndarray} for right subplot
    - k_components: PCA components (must be 1 for 1D KDE)
    - save_path: optional path to save figure as PDF
    """
    assert k_components == 1, "Only k_components=1 is supported for density plot"

    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 30}
    matplotlib.rc('font', **font)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4), sharey=False)

    def process_subplot(ax, dataset, title):
        arrays = {label: np.asarray(data) for label, data in dataset.items()}
        max_dim = max(arr.shape[1] for arr in arrays.values())
        aligned = {label: (np.pad(arr, ((0, 0), (0, max_dim - arr.shape[1])), mode='constant')
                           if arr.shape[1] < max_dim else arr[:, :max_dim]) for label, arr in arrays.items()}
        all_data = np.vstack(list(aligned.values()))
        pca = PCA(n_components=k_components)
        pca.fit(all_data)
        projections = {label: pca.transform(arr)[:, 0] for label, arr in aligned.items()}

        all_proj_flat = np.concatenate(list(projections.values()))
        margin = 0.01 * (all_proj_flat.max() - all_proj_flat.min())
        x_min = all_proj_flat.min() - margin
        x_max = all_proj_flat.max() + margin

        for label, proj in projections.items():
            sns.kdeplot(proj, label=label, fill=True, linewidth=1, cut=0.1, ax=ax)

        # ax.set_title(title)
        ax.set_xlabel("PCA Projection")
        ax.set_xlim(x_min, x_max)
        ax.set_xticks([-1, 0, 1])


    process_subplot(ax1, dataset_list1, "Group 1")
    process_subplot(ax2, dataset_list2, "Group 2")
    ax1.legend(fontsize=25, loc='upper right')
    ax2.legend(fontsize=18, loc='upper right')
    ax1.set_title("CICIDS2017")
    ax2.set_title("CICDDoS2019")
    ax1.set_xticks([0, 2, 4])
    ax1.set_yticks([0, 1, 2, 3, 4, 5])
    ax2.set_xticks([-1, 0, 1])
    ax2.set_yticks([0, 0.5, 1, 1.5])
    ax1.set_ylabel("Density")
    ax2.set_ylabel("")

    plt.subplots_adjust(wspace=0.15)
    matplotlib.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts
    matplotlib.rcParams['ps.fonttype'] = 42
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Saved plot to {save_path}")
    plt.show()




def cosine_similarity_pca_projection(datasets: dict, k_components=1):
    """
    Compute cosine similarity between datasets after PCA projection.

    Parameters:
    - datasets: dict of {label: np.ndarray}
    - k_components: number of PCA components to use (e.g., 1 or 2)

    Returns:
    - similarity_matrix: pd.DataFrame of cosine similarities
    """
    import pandas as pd

    # Convert values to NumPy arrays
    arrays = {label: np.asarray(data) for label, data in datasets.items()}
    all_data = np.vstack(list(arrays.values()))

    # PCA fit on combined data
    pca = PCA(n_components=k_components)
    pca.fit(all_data)

    # Project all datasets
    projections = {label: pca.transform(data) for label, data in arrays.items()}

    # Compute mean vector for each projected dataset
    mean_vectors = {label: np.mean(proj, axis=0) for label, proj in projections.items()}

    # Cosine similarity matrix
    labels = list(mean_vectors.keys())
    vectors = np.vstack([mean_vectors[label] for label in labels])
    cosine_sim = cosine_similarity(vectors)

    return pd.DataFrame(cosine_sim, index=labels, columns=labels)


def wasserstein_distance_direct(datasets: dict):
    """
    Compute pairwise Wasserstein distances between datasets directly (1D only).

    Parameters:
    - datasets: dict of {label: 1D np.ndarray or pd.Series}

    Returns:
    - pd.DataFrame of pairwise Wasserstein distances
    """
    # Ensure all inputs are 1D arrays
    arrays = {label: np.asarray(data).ravel() for label, data in datasets.items()}
    labels = list(arrays.keys())
    n = len(labels)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = wasserstein_distance(arrays[labels[i]], arrays[labels[j]])

    return pd.DataFrame(matrix, index=labels, columns=labels)


def compute_mmd(X, Y, gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two samples using RBF kernel.

    Parameters:
    - X, Y: np.ndarray datasets (n_samples, n_features)
    - gamma: RBF kernel coefficient (1 / (2 * sigma^2))

    Returns:
    - float: MMD distance
    """
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    mmd_squared = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return np.sqrt(max(mmd_squared, 0))  # ensure non-negative


# Maximum Mean Discrepancy (MMD) between pairs of datasets using a Gaussian (RBF) kernel
def mmd_distance_matrix(datasets: dict, gamma=1.0):
    """
    Compute pairwise MMD distances for a dictionary of datasets.

    Parameters:
    - datasets: dict of {label: np.ndarray}
    - gamma: RBF kernel parameter

    Returns:
    - pd.DataFrame of MMD distances
    """
    labels = list(datasets.keys())
    n = len(labels)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = compute_mmd(datasets[labels[i]], datasets[labels[j]], gamma)

    return pd.DataFrame(matrix, index=labels, columns=labels)




if __name__ == "__main__":
    # -------------------Data distribution plots---------------------------------------
    # UNSW_data_distribution_plots_V2()
    # CICIDS_data_distribution_plots()
    # CICDDoS2019_data_distribution_plots()
    # combined_distribution_plots_tight()
    # sys.exit()
    #--------------------PCA density plots--------------------------------------
    # Read datasets
    dataset_name = "CICIDS2017"
    dataset_name1 = "CICDDoS2019"
    # First dataset loading
    print("Dataset:", dataset_name)
    data_path = "Data&Model/"+dataset_name+"/"
    train_data_path = data_path+"training_all_classes.csv"
    ctgan_data_path = data_path+"ctgan_synthetic_data_all.csv"
    ddpm_data_path = data_path+"ddpm_synthetic_data_all.csv"
    great_data_path = data_path+"great_synthetic_data_all.csv"
    train_data = pd.read_csv(train_data_path)
    ctgan_data = pd.read_csv(ctgan_data_path)
    ddpm_data = pd.read_csv(ddpm_data_path)
    great_data = pd.read_csv(great_data_path)
    print(train_data.shape)
    print(ctgan_data.shape)
    print(ddpm_data.shape)
    print(great_data.shape)
    print(ctgan_data.columns)
    # Convert to numpy
    x_train = train_data.iloc[:, :-1].values
    x_ctgan = ctgan_data.iloc[:, :-1].values
    x_ddpm = ddpm_data.iloc[:, :-1].values
    x_great = great_data.iloc[:, :-1].values

    # second dataset loading
    print("Dataset1:", dataset_name1)
    data_path1 = "Data&Model/" + dataset_name1 + "/"
    train_data_path1 = data_path1 + "training_all_classes.csv"
    ctgan_data_path1 = data_path1 + "ctgan_synthetic_data_all.csv"
    ddpm_data_path1 = data_path1 + "ddpm_synthetic_data_all.csv"
    great_data_path1 = data_path1 + "great_synthetic_data_all.csv"
    train_data1 = pd.read_csv(train_data_path1)
    ctgan_data1 = pd.read_csv(ctgan_data_path1)
    ddpm_data1 = pd.read_csv(ddpm_data_path1)
    great_data1 = pd.read_csv(great_data_path1)

    print(train_data1.shape)
    print(ctgan_data1.shape)
    print(ddpm_data1.shape)
    print(great_data1.shape)
    print(ctgan_data1.columns)
    # sys.exit()
    # Convert to numpy
    x_train1 = train_data1.iloc[:, :-1].values
    x_ctgan1 = ctgan_data1.iloc[:, :-1].values
    x_ddpm1 = ddpm_data1.iloc[:, :-1].values
    x_great1 = great_data1.iloc[:, :-1].values


    # plot_pca_density_from_dataframe(datasets={"Original": x_train, "CTGAN": x_ctgan, "TabDDPM": x_ddpm, "GReaT": x_great}, k_components=1, save_path="plots/"+dataset_name+"_comparison.pdf")
    plot_two_pca_density_figures_side_by_side(dataset_list1={"Original": x_train, "CTGAN": x_ctgan, "TabDDPM": x_ddpm, "GReaT": x_great}, dataset_list2={"Original": x_train1, "CTGAN": x_ctgan1, "TabDDPM": x_ddpm1, "GReaT": x_great1}, k_components=1, save_path="plots/two_data_PCA_comparison.pdf")
    # similarity_df = cosine_similarity_pca_projection(datasets={"Original": x_train, "CTGAN": x_ctgan, "TabDDPM": x_ddpm, "GReaT": x_great}, k_components=5)
    # print(similarity_df)
    # wasserstein_distance = wasserstein_distance_direct(datasets={"Original": x_train, "CTGAN": x_ctgan, "TabDDPM": x_ddpm, "GReaT": x_great})
    # print(wasserstein_distance)
    # mmd_distance = mmd_distance_matrix(datasets={"Original": x_train, "CTGAN": x_ctgan, "TabDDPM": x_ddpm, "GReaT": x_great}, gamma=0.5)
    # print(mmd_distance)