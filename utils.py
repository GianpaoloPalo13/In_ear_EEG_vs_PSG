
import os
import glob
import warnings
import math
import mne
import numpy as np
import pandas as pd
import antropy as ant
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import seaborn as sb
from statsmodels.stats import inter_rater
from scipy.io import loadmat
from scipy.signal import periodogram, welch
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.stats import (mode, mstats, skew, kurtosis, iqr, pearsonr, gaussian_kde, shapiro, ttest_ind, mannwhitneyu,
                         f_oneway, kruskal)
from sklearn.preprocessing import StandardScaler
from matplotlib.collections import LineCollection
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
warnings.filterwarnings('ignore')


########################################################################################################################
# ---------------------------------------------  Hypnogram-based approach  ---------------------------------------------
########################################################################################################################


def uniform_scorers(psg_scorer_path, gdk_scorer_path, save_path):
    # Function to uniform PSG and in-ear-EEG scorers i.e., merging N1, N2, and N3 stages into NREM and removing
    # Movement and Unknown labels
    # Input:
    # - psg_scorer_path: absolute path to the folder containing PSG scorers
    # - gdk_scorer_path: absolute path to the folder containing in-ear-EEG scorers
    # - save_path: absolute path where to save uniformed scorers
    # Output:
    # - uniform_save_path_psg: absolute path containing uniformed PSG scorers
    # - uniform_save_path_gdk: absolute path containing uniformed in-ear-EEG scorers
    # - epc_path: absolute path where removed epochs are stored (related to Movement and Unknown labels)

    uniform_save_path_psg = os.path.join(save_path, 'new_PSG_scorers')
    if not os.path.exists(uniform_save_path_psg):
        os.makedirs(uniform_save_path_psg)
    uniform_save_path_gdk = os.path.join(save_path, 'new_GDK_scorers')
    if not os.path.exists(uniform_save_path_gdk):
        os.makedirs(uniform_save_path_gdk)
    epc_path = os.path.join(save_path, 'Epochs_removed')
    if not os.path.exists(epc_path):
        os.makedirs(epc_path)

    subs = os.listdir(psg_scorer_path)
    scorers = os.listdir(os.path.join(psg_scorer_path, subs[0]))

    for sub in subs:

        uniform_save_path_psg_sub = os.path.join(uniform_save_path_psg, sub)
        if not os.path.exists(uniform_save_path_psg_sub):
            os.makedirs(uniform_save_path_psg_sub)
        uniform_save_path_gdk_sub = os.path.join(uniform_save_path_gdk, sub)
        if not os.path.exists(uniform_save_path_gdk_sub):
            os.makedirs(uniform_save_path_gdk_sub)

        to_remove = []
        for scorer in scorers:

            uniform_save_path_psg_sub_scorer = os.path.join(uniform_save_path_psg_sub, scorer)
            if not os.path.exists(uniform_save_path_psg_sub_scorer):
                os.makedirs(uniform_save_path_psg_sub_scorer)
            uniform_save_path_gdk_sub_scorer = os.path.join(uniform_save_path_gdk_sub, scorer)
            if not os.path.exists(uniform_save_path_gdk_sub_scorer):
                os.makedirs(uniform_save_path_gdk_sub_scorer)

            hyp_psg = loadmat(glob.glob(os.path.join(psg_scorer_path, sub, scorer) + '/*.mat')[0])['stages'].squeeze()
            to_remove.extend(np.argwhere((hyp_psg == 6) | (hyp_psg == 7)).flatten().tolist())
            hyp_gdk = loadmat(glob.glob(os.path.join(gdk_scorer_path, sub, scorer) + '/*.mat')[0])['stages'].squeeze()
            to_remove.extend(np.argwhere((hyp_gdk == 6) | (hyp_gdk == 7)).flatten().tolist())

        to_remove = list(set(to_remove))
        to_remove.sort()
        np.save(os.path.join(epc_path, sub + '.npy'), to_remove)

        for scorer in scorers:
            hyp_psg = loadmat(glob.glob(os.path.join(psg_scorer_path, sub, scorer) + '/*.mat')[0])['stages'].squeeze()
            to_keep = np.delete(np.arange(len(hyp_psg)), to_remove)
            new_hyp_psg = hyp_psg[to_keep]
            new_hyp_psg[(new_hyp_psg == 1) | (new_hyp_psg == 2) | (new_hyp_psg == 3) | (new_hyp_psg == 4)] = 1
            new_hyp_psg[new_hyp_psg == 5] = 2

            hyp_gdk = loadmat(glob.glob(os.path.join(gdk_scorer_path, sub, scorer) + '/*.mat')[0])['stages'].squeeze()
            new_hyp_gdk = hyp_gdk[to_keep]
            new_hyp_gdk[(new_hyp_gdk == 1) | (new_hyp_gdk == 2) | (new_hyp_gdk == 3) | (new_hyp_gdk == 4)] = 1
            new_hyp_gdk[new_hyp_gdk == 5] = 2

            np.save(os.path.join(uniform_save_path_psg_sub, scorer, sub + '.npy'), new_hyp_psg)
            np.save(os.path.join(uniform_save_path_gdk_sub, scorer, sub + '.npy'), new_hyp_gdk)

    return uniform_save_path_psg, uniform_save_path_gdk, epc_path


def inter_variability(scorer_path):
    # Function to analyse the inter-scorer variability among scorers referring to the scoring of the same signal
    # i.e., either PSG or in-ear-EEG
    # Input:
    # - scorer_path: absolute path pointing to the folder containing the sub-folders for all subjects with all the
    #                scorers (i.e., hypnograms)
    # Output:
    # - fleiss: Fleiss' kappa values among scorers, one value per subject

    # Listing all the subjects
    subj = os.listdir(scorer_path)
    # Defining an array for Fleiss' kappa values with dimension equal to the number of subjects
    fleiss = np.zeros(len(subj))

    # Cycle on subjects
    for ns, i_subj in enumerate(subj):
        scorers_folder = os.path.join(scorer_path, i_subj)

        scorers = np.concatenate([np.asarray(np.load(glob.glob(os.path.join(scorers_folder, scorer, '*.npy'))[0]))
                                 .reshape(-1, 1) for scorer in os.listdir(scorers_folder)], axis=1)

        # Computing Fleiss' kappa among scorers
        table, cath = inter_rater.aggregate_raters(scorers)
        fleiss[ns] = inter_rater.fleiss_kappa(table, method='fleiss')

    return fleiss


def intra_variability(psg_scorer_path, gdk_scorer_path):
    # Function to analyse the intra-scorer variability among hypnograms scored by the same clinician
    # i.e., PSG vs in-ear-EEG
    # Input:
    # - psg_scorer_path: absolute path pointing to the folder containing the sub-folders for all subjects with all the
    #                    PSG scorers (i.e., hypnograms)
    # - gdk_scorer_path: absolute path pointing to the folder containing the sub-folders for all subjects with all the
    #                    in-ear-EEG scorers (i.e., hypnograms)
    # Output:
    # - cohen: Cohen's kappa values among scorers, three values per subject (one for each scorer)

    # Listing all the subjects
    subj = os.listdir(psg_scorer_path)
    # Defining a matrix for Fleiss' kappa values with dimensions [Number of scorers X Number of subjects]
    cohen = np.zeros([len(os.listdir(os.path.join(psg_scorer_path, subj[0]))), len(subj)])

    # Cycle on subjects
    for ns, i_subj in enumerate(subj):
        scorers_folder = os.path.join(psg_scorer_path, i_subj)
        for n_sc, scorer in enumerate(os.listdir(scorers_folder)):

            psg_scorer = np.asarray(np.load(os.path.join(scorers_folder, scorer, i_subj + '.npy'))).squeeze()
            gdk_scorer = np.asarray(np.load(os.path.join(scorers_folder, scorer, i_subj + '.npy').replace(
                os.path.basename(psg_scorer_path), os.path.basename(gdk_scorer_path)))).squeeze()

            # Computing Cohen's kappa among scorers
            cohen[n_sc, ns] = skm.cohen_kappa_score(psg_scorer, gdk_scorer)

    return cohen


def plot_inter(psg_fleiss, gdk_fleiss, save_path):
    # Function to plot the inter-scorer variability both for PSG and in-ear-EEG scorers
    # Input:
    # - psg_fleiss: Fleiss' kappa values for PSG scorers
    # - gdk_fleiss: Fleiss' kappa values for in-ear-EEG scorers
    # Output:
    # - the image representing the boxplot distributions for Fleiss' kappa values for PSG and in-ear-EEG scorers is
    #   saved in save_path

    color_gdk = np.array([64 / 255, 105 / 255, 229 / 255, 100 / 255])
    color_psg = np.array([240 / 255, 128 / 255, 126 / 255, 150 / 255])
    fig, axes = plt.subplots(figsize=(14, 7))
    axes.yaxis.grid(True)
    flier_props = {'marker': 'o', 'markersize': 13, 'markeredgecolor': 'red', 'markerfacecolor': 'white',
                   "markeredgewidth": 2}
    mean_props = {"marker": "+", "markeredgecolor": 'k', "markersize": 11, "markeredgewidth": 1.5}
    median_props = {"color": 'k', 'linewidth': 1.5}

    box0 = axes.boxplot(gdk_fleiss, positions=[0], patch_artist=True, showmeans=True,
                        meanprops=mean_props, medianprops=median_props, flierprops=flier_props,
                        boxprops=dict(facecolor=color_gdk))
    box1 = axes.boxplot(psg_fleiss, positions=[.5], patch_artist=True, showmeans=True,
                        meanprops=mean_props, medianprops=median_props, flierprops=flier_props,
                        boxprops=dict(facecolor=color_psg))
    axes.set_ylim(0, 1)
    axes.set_ylabel("Fleiss' Kappa", labelpad=10, fontsize=20)
    fig.suptitle("Inter-scorer variability", fontsize=23)
    axes.legend([box0["boxes"][0], box1["boxes"][0]], ['In-ear-EEG-to-In-ear-EEG scorers', 'PSG-to-PSG scorers'],
                loc='lower left', fontsize=15, bbox_to_anchor=(0.01, 1.03, .98, .103), ncols=2,
                mode="expand", borderaxespad=-0.5)
    fig.tight_layout(pad=1)
    plt.xticks([])
    plt.tick_params(axis="y", labelsize=14)
    fig.savefig(os.path.join(save_path, 'Inter_Scorer.jpg'), dpi=300)
    # plt.show(block=False)
    # plt.pause(5)
    plt.close()


def plot_intra(psg_gdk_cohen, save_path):
    # Function to plot the intra-scorer variability both for PSG and in-ear-EEG scorers
    # Input:
    # - psg_gdk_cohen: Cohen's kappa values for PSG and in-ear-EEG scorers
    # Output:
    # - the image representing the boxplot distributions for Cohen's kappa values for PSG and in-ear-EEG scorers is
    #   saved in save_path

    color = np.array([46 / 255, 66 / 255, 84 / 255, 50 / 255])
    fig, axes = plt.subplots(figsize=(14, 7))
    axes.yaxis.grid(True)
    mean_props = {"marker": "+", "markeredgecolor": 'k', "markersize": 13, "markeredgewidth": 1.5}
    median_props = {"color": 'k', 'linewidth': 1.5}
    flier_props = {'marker': 'o', 'markersize': 13, 'markeredgecolor': 'red', 'markerfacecolor': 'white',
                   "markeredgewidth": 2}

    box = axes.boxplot([psg_gdk_cohen[0, :], psg_gdk_cohen[1, :], psg_gdk_cohen[2, :]], positions=[0, 1, 2],
                       patch_artist=True, showmeans=True, meanprops=mean_props, medianprops=median_props,
                       flierprops=flier_props, boxprops=dict(facecolor=color))
    axes.set_ylim(0, 1)
    axes.set_ylabel("Cohen's Kappa", labelpad=10, fontsize=20)
    axes.set_title("Intra-scorer variability", fontsize=23)
    names = ['PSG-to-In-ear-EEG scorer 1', 'PSG-to-In-ear-EEG scorer 2', 'PSG-to-In-ear-EEG scorer 3']
    axes.set_xticks(np.arange(0, len(names)))
    axes.set_xticklabels(names)
    # axes.grid(True)
    fig.tight_layout(pad=1)
    plt.tick_params(axis='both', labelsize=14)
    fig.savefig(os.path.join(save_path, 'Intra_Scorer.jpg'), dpi=300)
    # plt.show(block=False)
    # plt.pause(5)
    plt.close()


def soft_agreement(scorers_folder):
    # Function to compute the soft-agreement metric for all scorers
    # Input:
    # - scorers_folder: absolute path pointing to the folder for a subject containing the sub-folders with the scorers
    # Output:
    # - metrics: dictionary having the scorer names as keys and the soft-agreement metrics as values

    # Loading the scorers
    scorers = {scorer: np.asarray(np.load(glob.glob(os.path.join(scorers_folder, scorer) + '/*.npy')[0]))
               for scorer in os.listdir(scorers_folder)}

    metrics = {}
    # Cycle on scorers
    for target in os.listdir(scorers_folder):
        hyp = scorers[target]
        other_hyps = [scorers[others] for others in os.listdir(scorers_folder) if others != target]
        # stages = number of classes i.e., sleep stages + 1
        # Working with Awake, NREM, and REM states --> stages = 3 + 1
        stages = 4
        epochs = range(len(hyp))
        prob_c = np.zeros((stages, len(hyp)))
        for other_hyp in other_hyps:
            prob_c[np.array(other_hyp) + 1, epochs] += 1
        prob_c_norm = prob_c / prob_c.max(0)
        soft_agr = prob_c_norm[np.array(hyp) + 1, epochs].mean()
        metrics[target] = soft_agr

    return metrics


def plot_table(data, x_names, y_names, title, save_path):
    # Function to represent Soft-agreement values in a table
    # Input:
    # - data: values to put in the table
    # - x_names: row labels for the table
    # - y_names: column labels for the table
    # - title: title to set for the table
    # - save_path: absolute path where to save the image
    # Output:
    # - the table is saved in save_path

    df = pd.DataFrame(data, index=x_names, columns=y_names)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(title, fontsize=23, fontweight='bold')
    ax.axis('off')
    tab = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(16)
    for key, cell in tab.get_celld().items():
        if key[0] == 0 or key[1] == -1:
            cell.set_text_props(weight='bold')
        cell.set_height(0.08)
    if 'PSG' in title: name = 'Soft_agreement_PSG'
    else: name = 'Soft_agreement_GDK'
    fig.savefig(os.path.join(save_path, name + '.jpg'), dpi=300)
    # plt.show(block=False)
    # plt.pause(5)
    plt.close()


def consensus_definition(scorer_path, save_path, to_plot=False, plot_path=None):
    # Function to define the consensus of the scorers
    # Input:
    # - scorer_path: absolute path pointing to the folder containing the sub-folders for all subjects with all the
    #                scorers (i.e., hypnograms)
    # - to_plot: if True, Soft-agreement values are represented in a table and saved in plot_path
    # - plot_path: absolute path where to save the plot for Soft-agreement values
    # - save_path: absolute path of the folder where to store the results
    # Output:
    # - save_path_cons: absolute path to the folder containing the consensus

    # Listing all the subjects
    subj = os.listdir(scorer_path)
    # Matrix to save soft-agreement values
    soft_matrix = np.zeros([len(subj), len(os.listdir(os.path.join(scorer_path, subj[0])))])

    if 'PSG' in os.path.split(scorer_path)[1]: sub_fold = 'PSG_consensus'; title = 'Soft-Agreement for PSG scorers'
    else: sub_fold = 'GDK_consensus';  title = 'Soft-Agreement for in-ear-EEG scorers'
    save_path_cons = os.path.join(save_path, sub_fold)
    if not os.path.exists(save_path_cons): os.makedirs(save_path_cons)

    # Cycle on subjects
    for ns, i_subj in enumerate(subj):
        scorers_folder = os.path.join(scorer_path, i_subj)
        # Computing soft-agreement metric
        metrics = soft_agreement(scorers_folder)
        soft_matrix[ns, :] = np.round(np.fromiter(metrics.values(), dtype=float), 4)

        # The most reliable scorer is the one associated with the highest soft-agreement value
        best_scorer = max(metrics, key=metrics.get)

        stages = np.concatenate([np.asarray(np.load(glob.glob(os.path.join(scorers_folder, scorer, '*.npy'))[0]))
                                .squeeze().reshape(-1, 1) for scorer in os.listdir(scorers_folder)], axis=1)
        # The consensus is defined by the sleep stages scored by the majority of the scorers
        consensus = mode(stages, keepdims=True, axis=1)
        # In case of ties, the mode is 1 --> we assign the sleep stages of the most reliable scorer
        ties = np.where(consensus[1] == 1)[0]
        if ties.any(): consensus[0][ties] = np.asarray(np.load(glob.glob(
            os.path.join(scorers_folder, best_scorer, '*.npy'))[0])).squeeze().reshape(-1, 1)[ties]

        # The name is equal to the corresponding subject
        np.save(os.path.join(save_path_cons, i_subj + '.npy'), consensus[0])

    if plot_path is None: plot_path = save_path
    if to_plot: plot_table(soft_matrix, subj, os.listdir(os.path.join(scorer_path, subj[0])), title, plot_path)

    return save_path_cons


def intersection(psg_consensus_path, gdk_consensus_path, save_path, consensus_agreement=True):
    # Function to define the intersection between PSG and in-ear-EEG consensus
    # Input:
    # - psg_consensus_path: absolute path pointing to the folder containing the PSG consensus for all subjects
    # - gdk_consensus_path: absolute path pointing to the folder containing the in-ear-EEG consensus for all subjects
    # - save_path: absolute path of the folder where to store the results
    # - consensus_agreement: if True, agreement metrics are evaluated between PSG and in-ear-EEG consensus i.e.,
    #                        precision, recall, and F1-score
    # Output:
    # - save_path_inter: absolute path to the folder containing the intersection of the two consensus

    # Loading consensus
    psg_files = glob.glob(psg_consensus_path + '/*.npy')
    gdk_files = glob.glob(gdk_consensus_path + '/*.npy')

    save_path_inter = os.path.join(save_path, 'Intersection')
    if not os.path.exists(save_path_inter): os.makedirs(save_path_inter)
    # The name is equal to the corresponding subject

    classes = np.unique(np.load(psg_files[0]).squeeze())
    metrics = {'precision': {cls: [] for cls in classes},
               'recall': {cls: [] for cls in classes},
               'f1-score': {cls: [] for cls in classes}}

    # Cycle on subjects
    for n_sbj in range(len(psg_files)):
        psg = np.load(psg_files[n_sbj]).squeeze()
        gdk = np.load(gdk_files[n_sbj]).squeeze()

        if consensus_agreement:
            report = skm.classification_report(psg, gdk, output_dict=True)
            for nc in range(len(metrics.keys())):
                for m in list(metrics.keys()):
                    metrics[m][nc].append(report[str(nc)][m])

        # epochs where the PSG and in-ear-EEG consensus do not agree are scored as 8
        inter_cons = np.where(psg == gdk, psg, 8 * np.ones_like(gdk))
        np.save(os.path.join(save_path_inter, os.path.split(psg_files[n_sbj])[1]), inter_cons)

    if consensus_agreement:
        print('\nEvaluating the agreement between consensus')
        for nc in range(len(metrics.keys())):
            for m in list(metrics.keys()):
                print(f'Average {m.capitalize()} for class {nc}: {np.mean(metrics[m][nc]):.4f} '
                      f'\u00B1 {np.std(metrics[m][nc]):.4f}')

    return save_path_inter


########################################################################################################################
# ----------------------------------------------  Feature-based approach  ----------------------------------------------
########################################################################################################################

# ------------------------------------------------  Feature extraction  ------------------------------------------------

def spectral(f, psd, psd_norm):
    # Function to extract:
    # - spectral energy (absolute power);
    # - relative power bands i.e., delta (D), theta (T), alpha (A), sigma (S), beta (B)
    # - relative power ratios i.e., delta/theta, delta/sigma, delta/beta, theta/alpha, delta/alpha, alpha/beta,
    #                               delta/(alpha + beta), theta/(alpha + beta), delta/(alpha + beta + theta)

    bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 16, 'Sigma'), (16, 30, 'Beta'),
             (30, 35, 'Gamma')]

    # Spectral Energy
    spec_en = sum(abs(psd_norm))                                # Depends on signal amplitude

    # Relative Spectral Powers
    p_rel = np.zeros(6)
    for band in range(len(bands)):
        ind_band = np.where((f > bands[band][0]) & (f <= bands[band][1]))
        p_rel[band] = sum(abs(psd[ind_band[0]])) / sum(abs(psd))

    # Spectral Power Ratios
    p_rat = np.zeros(9)                                         # Power Ratios
    p_rat[0] = p_rel[0] / p_rel[1]                              # D/T
    p_rat[1] = p_rel[0] / p_rel[3]                              # D/S
    p_rat[2] = p_rel[0] / p_rel[4]                              # D/B
    p_rat[3] = p_rel[1] / p_rel[2]                              # T/A
    p_rat[4] = p_rel[0] / p_rel[2]                              # D/A
    p_rat[5] = p_rel[2] / p_rel[4]                              # A/B
    p_rat[6] = p_rel[0] / (p_rel[2] + p_rel[4])                 # D/A+B
    p_rat[7] = p_rel[1] / (p_rel[2] + p_rel[4])                 # T/A+B
    p_rat[8] = p_rel[0] / (p_rel[2] + p_rel[4] + p_rel[1])      # D/A+B+T

    return spec_en, p_rel, p_rat


def renyi_entropy_func(x, sf, method="fft", nperseg=None, normalize=False, axis=-1):
    # Function to compute Renyi's entropy
    # Input:
    # - x: signal
    # - sf: sampling frequency, [Hz]

    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == "fft":
        _, psd = periodogram(x, sf, axis=axis)
    elif method == "welch":
        _, psd = welch(x, sf, nperseg=nperseg, axis=axis)

    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    renyi = -np.log((psd_norm ** 2).sum(axis=axis)) / np.log(2)

    if normalize:
        renyi /= np.log2(psd_norm.shape[axis])

    return renyi


def roll_off(frequency, psd):
    # Function to compute spectral roll-off i.e., the frequency below which there is the 85% of the spectrum's energy
    # Input:
    # - frequency: array of frequencies, [Hz]
    # - psd: PSD of the signal

    ro_ind = 0
    ro_sum = 0
    for i in range(0, len(psd)):
        ro_sum = ro_sum + abs(psd[i])
        if ro_sum > (0.85 * np.sum(abs(psd))):
            ro_ind = i
            break

    return frequency[ro_ind]


def features(x_sig, fs, no_epc, period=30, win_sec=5):
    # Function to extract the features
    # Input:
    # - x_sig: array of data
    # - fs: sampling frequency, [Hz]
    # - no_epc: indexes of epochs to remove
    # - period: epoch length, [s]
    # - win_sec: window length, [s]
    #            Note: win_sec should be at least double of the inverse of the lowest frequency of interest
    #                  having f_min = 0.5 Hz, win_sec is set to 2*(1/0.5) = 4 s (5 s to be conservative)

    # Ordered list of all the extracted features
    feats = ['Spectral energy', 'Relative delta power band', 'Relative theta power band',
             'Relative alpha power band', 'Relative sigma power band', 'Relative beta power band',
             'Relative gamma power band', 'delta-theta power ratio', 'delta-sigma power ratio',
             'delta-beta power ratio', 'theta-alpha power ratio', 'delta-alpha power ratio',
             'alpha-beta power ratio', 'delta-(alpha-beta) power ratio', 'theta-(alpha-beta) power ratio',
             'delta-(alpha-beta-theta) power ratio', 'Spectral centroid', 'Spectral crest factor',
             'Spectral flatness', 'Spectral skewness', 'Spectral kurtosis', 'Spectral mean', 'Spectral variance',
             'Spectral rolloff', 'Spectral spread', 'Standard deviation', 'Inter-quartile range', 'Skewness',
             'Kurtosis', 'Number of zero-crossings', 'Maximum first derivative', 'Hjorth activity',
             'Hjorth mobility', 'Hjorth complexity', 'Spectral entropy', 'Renyi entropy', 'Approximate entropy',
             'Sample entropy', 'Singular value decomposition entropy', 'Permutation entropy',
             'De-trended fluctuation analysis', 'Katz fractal dimension', 'Higuchi fractal dimension',
             'Petrosian fractal dimension', 'Lempel–Ziv complexity']

    # Number of epochs
    nepc = int(math.ceil(np.size(x_sig) / (period * int(fs))))
    # Defining a matrix containing feature values for all the epochs
    # --> Dimensions [N_features X N_epochs]
    features_matrix = np.zeros([len(feats), nepc - len(no_epc)])
    # Counter to fill the feature matrix
    count = 0

    # Storing the maximum of the signal to normalize recordings to extract features depending on amplitude
    max_s = np.max(abs(x_sig))

    # Cycle on epochs
    for time in range(nepc):
        if time not in no_epc:
            # Isolating each epoch i.e., 30-second samples of signals
            x_sig_30 = np.array(x_sig)[time * period * int(fs): (time + 1) * period * int(fs)]

            # performing zero-padding if the last epoch is smaller than the others
            if time == nepc - 1:
                add = np.zeros([period * int(fs) - np.size(x_sig_30, 0), 1])
                x_sig_30 = (np.concatenate([x_sig_30.reshape(-1, 1), add])).squeeze()

            # Mean value removal to compute the PSD
            x_sig_30_mean = x_sig_30 - np.mean(x_sig_30)
            # assessing PSD using Welch method
            f, psd = welch(x=x_sig_30_mean, fs=int(fs), window='hamming',
                           nperseg=int(win_sec * int(fs)), average='median')
            # assessing PSD on normalized signal for measuring features depending on amplitude
            f_norm, psd_norm = welch(x=(x_sig_30_mean / max_s), fs=int(fs), window='hamming',
                                     nperseg=int(win_sec * int(fs)), average='median')

            ########################################################################################
            # ----------------------------  Frequency-domain features  -----------------------------
            ########################################################################################
            spec_en, p_rel, p_rat = spectral(f, psd, psd_norm)

            # Spectral energy
            features_matrix[0, count] = spec_en

            # Relative spectral powers
            for nf in range(np.size(p_rel)):
                features_matrix[nf + 1, count] = p_rel[nf]

            # Spectral power ratios
            for nf in range(np.size(p_rat)):
                features_matrix[nf + 7, count] = p_rat[0]

            # Spectral centroid
            features_matrix[16, count] = np.sum(f * abs(psd)) / np.sum(abs(psd))

            # Spectral crest factor
            features_matrix[17, count] = np.max(abs(psd)) / np.mean(abs(psd))

            # Spectral flatness
            features_matrix[18, count] = mstats.gmean(abs(psd)) / np.mean(abs(psd))

            # Spectral skewness
            features_matrix[19, count] = skew(abs(psd))

            # Spectral kurtosis
            features_matrix[20, count] = kurtosis(abs(psd))

            # Spectral mean
            features_matrix[21, count] = np.mean(abs(psd_norm))                       # Depends on signal amplitude

            # Spectral variance
            features_matrix[22, count] = np.var(abs(psd_norm))                        # Depends on signal amplitude

            # Spectral roll-off
            features_matrix[23, count] = roll_off(f, psd)

            # Spectral spread
            features_matrix[24, count] = np.sum(((f - features_matrix[16, count]) ** 2) * abs(psd)) / np.sum(abs(psd))

            ########################################################################################
            # -------------------------------  Time-domain features  -------------------------------
            ########################################################################################
            # Standard Deviation
            features_matrix[25, count] = np.std(x_sig_30 / max_s)                     # Depends on signal amplitude

            # Inter-quartile range
            features_matrix[26, count] = iqr(x_sig_30 / max_s)                        # Depends on signal amplitude

            # Skewness
            features_matrix[27, count] = skew(x_sig_30)

            # Kurtosis
            features_matrix[28, count] = kurtosis(x_sig_30)

            # Number of zero-crossings
            features_matrix[29, count] = ant.num_zerocross(x_sig_30)

            # Maximum of the first derivative
            features_matrix[30, count] = np.max(np.diff(x_sig_30 / max_s, axis=-1))   # Depends on signal amplitude

            # Hjorth parameters i.e., activity, mobility, complexity
            features_matrix[31, count] = np.var(x_sig_30 / max_s)                     # Depends on signal amplitude
            features_matrix[32, count], features_matrix[33, count] = ant.hjorth_params(x_sig_30)

            # Spectral/Shannon entropy
            features_matrix[34, count] = ant.spectral_entropy(x_sig_30, sf=int(fs), method='welch',
                                                              nperseg=int(win_sec * int(fs)), normalize=True)

            # Renyi's entropy
            features_matrix[35, count] = renyi_entropy_func(x_sig_30, sf=int(fs), method='welch',
                                                            nperseg=int(win_sec * int(fs)), normalize=True)

            # Approximate entropy
            features_matrix[36, count] = ant.app_entropy(x_sig_30)

            # Sample entropy
            features_matrix[37, count] = ant.sample_entropy(x_sig_30)

            # Singular Value Decomposition entropy
            features_matrix[38, count] = ant.svd_entropy(x_sig_30, normalize=True)

            # Permutation entropy
            features_matrix[39, count] = ant.perm_entropy(x_sig_30, normalize=True)

            # De-trended fluctuation analysis exponent
            features_matrix[40, count] = ant.detrended_fluctuation(x_sig_30)

            # Fractal dimensions i.e., Katz, Higuchi, Petrosian FDs
            features_matrix[41, count] = ant.katz_fd(x_sig_30)
            features_matrix[42, count] = ant.higuchi_fd(x_sig_30.astype(np.float64))
            features_matrix[43, count] = ant.petrosian_fd(x_sig_30)

            # Lempel-Ziv complexity coefficient
            x_sig_30_bin = np.zeros(np.size(x_sig_30))
            x_sig_30_bin[np.argwhere(x_sig_30 > np.median(x_sig_30))[:, 0]] = 1
            features_matrix[44, count] = ant.lziv_complexity(x_sig_30_bin, normalize=True)

            count += 1

    return features_matrix


def feature_extraction(psg_data_path, gdk_data_path, epc_path, save_path, fs_psg=256, fs_gdk=250, period=30, win_sec=5):
    # Function to extract time- and frequency- domain features both from PSG and in-ear-EEG data
    # Input:
    # - psg_data_path: absolute path containing PSG data
    # - gdk_data_path: absolute path containing in-ear-EEG data
    # - epc_path: absolute path containing previously removed epochs (related to Movement and Unknown labels)
    # - save_path: absolute path where to save the extracted features
    # - fs_psg: PSG sampling frequency, [Hz]
    # - fs_gdk: in-ear-EEG sampling frequency, [Hz]
    # - period: epoch length, [s]
    # - win_sec: window length, [s]
    #            Note: win_sec should be at least double of the inverse of the lowest frequency of interest
    #                  having f_min = 0.5 Hz, win_sec is set to 2*(1/0.5) = 4 s (5 s to be conservative)
    # Output:
    # - feat_path: absolute path to the folder containing the features

    # Defining subject labels
    n_sbj = len(glob.glob(psg_data_path + '/*.mat'))                                # Number of subjects
    sbj_names = ['Subject_' + f"{(n + 1):02d}" for n in range(n_sbj)]

    feat_path = os.path.join(save_path, 'Features')
    if not os.path.exists(feat_path):
        os.mkdir(feat_path)

    to_remove = glob.glob(epc_path + '/*.npy')

    # Loading PSG data
    for nf, file in enumerate(glob.glob(psg_data_path + '/*.mat')):
        mat_file = loadmat(file)
        no_epc = np.load(to_remove[nf]).squeeze()

        # Defining a sub-folder for each subject where to save the features
        feat_path_sbj = os.path.join(feat_path, sbj_names[nf])
        if not os.path.exists(feat_path_sbj):
            os.mkdir(feat_path_sbj)

        eeg_bi = mat_file['EEG_bi']
        eeg_bi_names = mat_file['EEG_bi_names'].tolist()
        for ch in range(len(eeg_bi_names)):
            # Feature extraction
            feats = features(eeg_bi[ch, :], fs_psg, no_epc, period=period, win_sec=win_sec)
            np.save(os.path.join(feat_path_sbj, 'EEG_bi_' + eeg_bi_names[ch] + '.npy'), feats)

        eog_bi = mat_file['EOG_bi']
        eog_bi_names = mat_file['EOG_bi_names'].tolist()
        for ch in range(len(eog_bi_names)):
            # Feature extraction
            feats = features(eog_bi[ch, :], fs_psg, no_epc, period=period, win_sec=win_sec)
            np.save(os.path.join(feat_path_sbj, 'EOG_bi_' + eog_bi_names[ch] + '.npy'), feats)

        eeg_uni = mat_file['EEG_uni']
        eeg_uni_names = mat_file['EEG_uni_names'].tolist()
        for ch in range(len(eeg_uni_names)):
            # Feature extraction
            feats = features(eeg_uni[ch, :], fs_psg, no_epc, period=period, win_sec=win_sec)
            np.save(os.path.join(feat_path_sbj, 'EEG_uni_' + eeg_uni_names[ch] + '.npy'), feats)

        # Definition of the mastoid-to-mastoid derivation i.e., M2-M1
        m2m1 = (mat_file['EEG_uni'][mat_file['EEG_uni_names'].tolist().index('M2'), :] -
                mat_file['EEG_uni'][mat_file['EEG_uni_names'].tolist().index('M1'), :])
        feats = features(m2m1, fs_psg, no_epc, period=period, win_sec=win_sec)
        np.save(os.path.join(feat_path_sbj, 'EEG_bi_M2M1.npy'), feats)

        eog_uni = mat_file['EOG_uni']
        eog_uni_names = mat_file['EOG_uni_names'].tolist()
        for ch in range(len(eog_uni_names)):
            # Feature extraction
            feats = features(eog_uni[ch, :], fs_psg, no_epc, period=period, win_sec=win_sec)
            np.save(os.path.join(feat_path_sbj, 'EOG_uni_' + eog_uni_names[ch] + '.npy'), feats)

        print('Subject ' + str(nf + 1) + ': features extracted from PSG channels')

    # Loading in-ear-EEG data
    for nf, file in enumerate(glob.glob(gdk_data_path + '/*.mat')):
        mat_file = loadmat(file)
        no_epc = np.load(to_remove[nf]).squeeze()

        # Defining a sub-folder for each subject where to save the features
        feat_path_sbj = os.path.join(feat_path, sbj_names[nf])

        ear = mat_file['GDK']
        ear_names = mat_file['GDK_name'].tolist()
        for ch in range(len(ear_names)):
            # Feature extraction
            feats = features(ear[ch, :], fs_gdk, no_epc, period=period, win_sec=win_sec)
            np.save(os.path.join(feat_path_sbj, 'GDK_' + ear_names[ch] + '.npy'), feats)

        print('Subject ' + str(nf + 1) + ': features extracted from GDK channel')

    return feat_path


# ------------------------------------------------  Handling NaN values  -----------------------------------------------

def handle_nan(feat_path, inter_path, flag=False, save_path=None):
    # Function to handle NaN values within the feature matrices
    # - There are too many NaN values for channel M2 for subjects 3 and 6, thus it gets completely removed
    # - One epoch for subjects 1 and 8 (last and first epochs, respectively) contains NaN values, thus they are removed
    # Input:
    # - feat_path: absolute path to the folder containing the feature matrices
    # - inter_path: absolute path to the folder containing the reference i.e., intersection among consensus
    # - flag: if True, matrices indicating positions for NaN values are saved
    # - save_path: absolute path where to save NaN matrices (if flag is True)

    if flag:

        save_path_nan = os.path.join(save_path, 'NaN values')
        if not os.path.exists(save_path_nan):
            os.makedirs(save_path_nan)

        subs = os.listdir(feat_path)
        for sub in subs:
            # Cycle on feature matrices
            for nf, file in enumerate(glob.glob(os.path.join(feat_path, sub) + '/*.npy')):
                feats = np.load(file).squeeze()
                if nf == 0:
                    # Defining a matrix with dimensions [Number of features X Number of epochs]
                    matrix_nan = np.zeros(np.shape(feats))
                    # List where to save the name of the channels having NaN values
                    nan_names = []
                idx_nan = np.argwhere(np.isnan(feats))
                if len(idx_nan) > 0:
                    nan_names.append(os.path.basename(file).split('.')[0])
                    for i in idx_nan:
                        # Increasing by one the cell related to a specific [feature, epoch] combination with NaN value
                        i_r, i_c = i
                        matrix_nan[i_r, i_c] += 1

            if len(np.where(matrix_nan != 0)) > 0:
                np.save(os.path.join(save_path_nan, sub + '_NaN.npy'), matrix_nan)

    # Handling NaN values for subjects 1 and 8
    sbj_s = ['Subject_01', 'Subject_08']
    for sbj in sbj_s:
        # There is just one compromised epoch for subjects 1 and 8
        # Note: the epoch must be removed also from the reference i.e., intersection between consensus

        # The index for the compromised epoch for Subject 1 is 488 (i.e., the 489th epoch)
        if sbj == 'Subject_01': idx = 488
        # The index for the compromised epoch for Subject 8 is 0 (i.e., the 1st epoch)
        else: idx = 0

        for nf, file in enumerate(glob.glob(os.path.join(feat_path, sbj) + '/*.npy')):
            feats = np.load(file).squeeze()
            new_feats = np.delete(feats, idx, axis=1)
            np.save(file, new_feats)

        ref = np.load(os.path.join(inter_path, sbj + '.npy')).squeeze()
        new_ref = np.delete(ref, idx)
        np.save(os.path.join(inter_path, sbj + '.npy'), new_ref)

    # Handling NaN values for subjects 3 and 6
    sbj_s = ['Subject_03', 'Subject_06']
    for sbj in sbj_s:
        # The unipolar derivation M2 for these two subjects is too compromised i.e., it shows too many NaN values
        # hence it gets removed
        os.remove(os.path.join(feat_path, sbj, 'EEG_uni_M2.npy'))


# -------------------------------------------------  Feature selection  ------------------------------------------------

def redundancy_rate(matrix):
    # Function to compute the redundancy rate, that is used to verify the choice for the best k-nearest neighbors value
    # --> the lower the value, the less the redundancy within the feature subset
    # The input matrix must show dimensions [N_samples x N_features]

    corr = 0
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.to_numpy(copy=True)
    for f_i in range(np.size(matrix, 1)):
        for f_j in range(f_i + 1, np.size(matrix, 1)):
            corr += abs(pearsonr(matrix[:, f_i], matrix[:, f_j])[0])

    rr = corr/(np.size(matrix, 1)*(np.size(matrix, 1) - 1))

    return rr


def representation_entropy(matrix):
    # Function to compute the representation entropy that is used to optimize the choice for k-nearest neighbors value
    # --> the higher the value, the less the redundancy within the feature subset
    # The input matrix must show dimensions [N_samples x N_features]

    w, _ = np.linalg.eigh(np.cov(matrix, rowvar=False))
    w = w / np.sum(w)
    hr = -np.nansum(w*np.log2(abs(w)))

    return hr


def mici(x, y):
    # Function to compute the MICI - Maximal Information Compression Index i.e., a similarity measure on which the
    # feature selection algorithm here implemented is based on
    # --> the lower the value, the higher the dependency between the features
    # x, y: input features arrays

    eigenvalue, _ = np.linalg.eig(np.cov(x, y))           # Covariance matrix definition
    lambda_2 = np.min(eigenvalue)                         # MICI = minimum eigenvalue

    # Normalization to not have sensitivity on different feature scales
    norm_mici = lambda_2/(np.var(x) + np.var(y))
    return norm_mici


def fsfs(o, k):
    # Function to perform unsupervised feature selection
    # Reference article: Pabitra Mitra, CA Murthy, and Sankar K. Pal. “Unsupervised feature selection using feature
    # similarity”. In: IEEE transactions on pattern analysis and machine intelligence 24.3 (2002), pp. 301–312.
    # doi: 10.1109/34.990133.
    # Note: this code was originally written by Giuliana Monachino.

    features_lbl = o.columns
    indices = o.index
    to_retain_lbl = []                                  # Selected features
    to_discard_lbl = []                                 # Removed features
    o = o.to_numpy()                                    # o: original feature subset
    r = o                                               # r: reduced feature subset. At first, r = o

    s = squareform(pdist(np.transpose(r), lambda p, q: mici(p, q)))             # s: dissimilarity matrix
    # --> 'pdist' computes distances between all pairs of arrays within 'r' according to the mici metric
    # --> 'squareform' is needed to get a square and symmetric matrix
    # E.g., working with four features, we get a 4x4 correlation matrix and 'pdist' returns 6 elements, in
    # positions (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3).

    s[range(s.shape[0]), range(s.shape[0])] = np.nan       # Setting all elements on the diagonal to 'NaN'
    # Sorting elements on each row in ascending order --> the lower the value, the higher the dependency
    sorted_s = np.sort(s, axis=1)
    # On rows, indexes of sorted elements i.e., indexes of the features with which each feature is correlated the most
    sorted_idx = np.argsort(s, axis=1)
    min_r = np.min(sorted_s, axis=0)

    # Definition of the error threshold (to adjust k-value during iterations): distance from the kth element of 'min_r'
    # i.e., the furthest element from the 'neighborhood'
    eps = min_r[k]

    # For convenience k = k + 1, such that writing matrix[0: k, :] we take all the rows from the first to the kth one
    # (including the kth row itself)
    k += 1

    stop = False
    while not stop:
        # Saving the index of the feature with the lowest dissimilarity from its kth element (to retain)
        to_retain_idx = np.argmin(sorted_s[:, k - 1])
        to_retain_lbl.append(features_lbl[to_retain_idx])       # Saving the name of the feature to retain

        # Saving the index of the k-nearest neighbors of the selected feature (to remove)
        to_discard_idx = sorted_idx[to_retain_idx, :k]
        to_discard_lbl.append(features_lbl[to_discard_idx])     # Saving the names of the features to discard

        r = np.delete(r, to_discard_idx, axis=1)
        features_lbl = features_lbl.delete(to_discard_idx)

        # Being 's' a symmetric matrix, features must be removed both by row and column
        s = np.delete(s, to_discard_idx, axis=0)
        s = np.delete(s, to_discard_idx, axis=1)

        # Following feature removal, the process repeats itself
        sorted_s = np.sort(s, axis=1)
        sorted_idx = np.argsort(s, axis=1)

        # If the k-value i.e. the number of neighbors to select for each feature is greater than the number of remaining
        # features, k gets decreased and set to the number of remaining features
        if k > r.shape[1] - 1: k = r.shape[1] - 1
        # If the k-value is decreased up to zero i.e., there is no neighbor left, then the code stops
        if k == 0: stop = True

        # We measure again the smallest distance (i.e., minimum dissimilarity) among all the remaining features
        # (between each feature and its corresponding kth element)
        r_k = np.min(sorted_s[:, k - 1])

        # Until the minimum distance is higher than the error threshold, k gets decreased by one
        while r_k > eps and not stop:
            k -= 1
            r_k = np.min(sorted_s[:, k - 1])

            if k == 0:
                stop = True
                break

    r = pd.DataFrame(r, index=indices, columns=features_lbl)
    return r, to_retain_lbl, to_discard_lbl


def k_optimization(dataset, k_values):
    # Function to optimize the number of k-nearest neighbours
    # The optimum value for k is found as the one that maximizes the representation entropy
    # A double check on the best k-value is performed i.e.,
    # Input:
    # - dataset: data to consider for optimizing k
    # - k_values: all k values to test

    # Reference value for optimization i.e., representation entropy computed on initial non-reduced feature subset
    ref_hr = representation_entropy(dataset.values)
    # Reference value for final check i.e., redundancy rate computed on initial non-reduced feature subset
    ref_rr = redundancy_rate(dataset.values)

    hr_values = []
    rr_values = []
    k_eff = []
    for k in k_values:
        r, to_retain_lbl, to_discard_lbl = fsfs(dataset, k)
        if np.size(r.values, 1) > 1:
            hr_values.append(representation_entropy(r.values))
            rr_values.append(redundancy_rate(r.values))
            k_eff.append(k)

    # Sorting the representation entropy values in descending order
    idx = np.argsort(hr_values)[::-1]
    hr_sort = [hr_values[i] for i in idx]
    rr_sort = [rr_values[i] for i in idx]
    k_sort = [k_eff[i] for i in idx]

    best_k = np.min(np.asarray(k_eff))
    for hr_val, rr_val, k_val in zip(hr_sort, rr_sort, k_sort):
        # Double check:
        # 1. the maximum representation entropy must be greater than the reference
        # 2. the minimum redundancy rate must be smaller than the reference
        if hr_val >= ref_hr and rr_val <= ref_rr:
            best_k = k_val
            return best_k

    return best_k


def feature_selection(inter_path, feat_path, flag):
    # Function to perform the feature selection
    # Input:
    # - inter_path: absolute path to the reference for the sleep stage analysis i.e., intersection between consensus
    # - feat_path: absolute path to the extracted features
    # - flag: if 'ear_vs_psg', the PSG-to-In-ear-EEG comparison analysis is performed; otherwise, the PSG-to-PSG
    #         comparison analysis takes place

    # Ordered list of all the extracted features
    feats = ['Spectral energy', 'Relative delta power band', 'Relative theta power band',
             'Relative alpha power band', 'Relative sigma power band', 'Relative beta power band',
             'Relative gamma power band', 'delta-theta power ratio', 'delta-sigma power ratio',
             'delta-beta power ratio', 'theta-alpha power ratio', 'delta-alpha power ratio',
             'alpha-beta power ratio', 'delta-(alpha-beta) power ratio', 'theta-(alpha-beta) power ratio',
             'delta-(alpha-beta-theta) power ratio', 'Spectral centroid', 'Spectral crest factor',
             'Spectral flatness', 'Spectral skewness', 'Spectral kurtosis', 'Spectral mean', 'Spectral variance',
             'Spectral rolloff', 'Spectral spread', 'Standard deviation', 'Inter-quartile range', 'Skewness',
             'Kurtosis', 'Number of zero-crossings', 'Maximum first derivative', 'Hjorth activity',
             'Hjorth mobility', 'Hjorth complexity', 'Spectral entropy', 'Renyi entropy', 'Approximate entropy',
             'Sample entropy', 'Singular value decomposition entropy', 'Permutation entropy',
             'De-trended fluctuation analysis', 'Katz fractal dimension', 'Higuchi fractal dimension',
             'Petrosian fractal dimension', 'Lempel–Ziv complexity']

    # Sleep stages analysed
    stages = ['Awake', 'NREM', 'REM']
    # Array with all the possible values for the k-nearest neighbours parameter
    k_values = np.arange(0, (len(feats) - 1), 1)

    save_path = os.path.join(os.path.split(feat_path)[0], 'Selected features')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path_flag = os.path.join(save_path, flag)
    if not os.path.exists(save_path_flag):
        os.mkdir(save_path_flag)

    # Listing all subjects
    subs = os.listdir(feat_path)
    # Listing all PSG files
    psg_chs_paths = [path for path in glob.glob(os.path.join(feat_path, subs[0]) + '/*.npy') if 'GDK' not in path]

    # ----------------------------------  PSG-to-In-ear-EEG comparison analysis  -----------------------------------
    if flag == 'ear_vs_psg':

        # Saving path to GDK file
        gdk_ch_path = glob.glob(os.path.join(feat_path, subs[0]) + '/GDK*.npy')[0]

        # Cycle on channels
        for psg_ch_path in psg_chs_paths:

            # Cycle on sleep stages
            for nc in range(len(stages)):

                save_path_stage = os.path.join(save_path_flag, stages[nc])
                if not os.path.exists(save_path_stage):
                    os.mkdir(save_path_stage)

                # Cycle on subjects
                for sub in subs:

                    jump = 0
                    # Channel M2 has been removed from subjects 3 and 6
                    if (sub in ['Subject_03', 'Subject_06']) & (os.path.basename(psg_ch_path).split('.')[0] ==
                                                                'EEG_uni_M2'): jump = 1
                    if jump == 0:
                        # Saving the correct file paths
                        psg_file = psg_ch_path.replace(subs[0], sub)
                        gdk_file = gdk_ch_path.replace(subs[0], sub)
                        # Loading the reference
                        ref_file = np.load(os.path.join(inter_path, sub + '.npy')).squeeze()

                        # Extracting the indexes for Awake (0), NREM (1), and REM (2) stages
                        ref = np.argwhere(ref_file == nc).squeeze()

                        # Loading PSG and in-ear-EEG values separately for each sleep stage
                        psg = np.load(psg_file).squeeze()[:, ref]
                        gdk = np.load(gdk_file).squeeze()[:, ref]

                        # Combining feature values from different subjects of each PSG channel with the in-ear-EEG
                        if sub == subs[0]: ds = psg.copy()
                        else: ds = np.concatenate([ds, psg], axis=1)
                        ds = np.concatenate([ds, gdk], axis=1)

                # Z-score standardization
                scaler = StandardScaler().fit(ds.transpose())
                ds_norm = scaler.transform(ds.transpose())
                ds_pd = pd.DataFrame(ds_norm, columns=feats)

                # Optimization of the number of k-nearest neighbours for the feature selection algorithm
                # The best k-value is found as the one related to the minimum of the representation entropy
                best_k = k_optimization(ds_pd, k_values)

                # Application of the FSFS algorithm
                r, to_retain_lbl, to_discard_lbl = fsfs(ds_pd, best_k)

                name = 'GDK_' + os.path.basename(psg_ch_path).split('.')[0].split('_')[-1] + '.npy'
                np.save(os.path.join(save_path_stage, name), r.columns)

    # --------------------------------------  PSG-to-PSG comparison analysis  --------------------------------------
    elif flag == 'psg_vs_psg':

        # Cycle on PSG channels i.e., two cycles in order to analyse all possible pairs of PSG derivations
        # if there are 21 PSG channels --> 210 possible pairs of PSG channels
        for psg_ind_1 in range(len(psg_chs_paths) - 1):
            for psg_ind_2 in np.arange(psg_ind_1 + 1, len(psg_chs_paths)):

                # Cycle on sleep stages
                for nc in range(len(stages)):

                    save_path_stage = os.path.join(save_path_flag, stages[nc])
                    if not os.path.exists(save_path_stage):
                        os.mkdir(save_path_stage)

                    # Cycle on subjects
                    for sub in subs:

                        jump = 0
                        # Channel M2 has been removed from subjects 3 and 6
                        if ((sub in ['Subject_03', 'Subject_06']) &
                                ((os.path.basename(psg_chs_paths[psg_ind_1]).split('.')[0] == 'EEG_uni_M2') |
                                 (os.path.basename(psg_chs_paths[psg_ind_2]).split('.')[0] == 'EEG_uni_M2'))): jump = 1

                        if jump == 0:
                            # Saving the correct file paths
                            psg_file_1 = psg_chs_paths[psg_ind_1].replace(subs[0], sub)
                            psg_file_2 = psg_chs_paths[psg_ind_2].replace(subs[0], sub)

                            # Loading the reference
                            ref_file = np.load(os.path.join(inter_path, sub + '.npy')).squeeze()

                            # Extracting the indexes for Awake (0), NREM (1), and REM (2) stages
                            ref = np.argwhere(ref_file == nc).squeeze()

                            # Loading PSG and in-ear-EEG values separately for each sleep stage
                            psg_1 = np.load(psg_file_1).squeeze()[:, ref]
                            psg_2 = np.load(psg_file_2).squeeze()[:, ref]

                            # Combining feature values from different subjects of each possible pair of PSG channels
                            if sub == subs[0]: ds = psg_1.copy()
                            else: ds = np.concatenate([ds, psg_1], axis=1)
                            ds = np.concatenate([ds, psg_2], axis=1)

                    # Z-score standardization
                    scaler = StandardScaler().fit(ds.transpose())
                    ds_norm = scaler.transform(ds.transpose())
                    ds_pd = pd.DataFrame(ds_norm, columns=feats)

                    # Optimization of the number of k-nearest neighbours for the feature selection algorithm
                    # The best k-value is found as the one related to the minimum of the representation entropy
                    best_k = k_optimization(ds_pd, k_values)

                    # Application of the FSFS algorithm
                    r, to_retain_lbl, to_discard_lbl = fsfs(ds_pd, best_k)

                    name = (os.path.basename(psg_chs_paths[psg_ind_1]).split('.')[0].split('_')[-1] + '_' +
                            os.path.basename(psg_chs_paths[psg_ind_2]).split('.')[0].split('_')[-1] + '.npy')
                    np.save(os.path.join(save_path_stage, name), r.columns)

    return save_path_flag


def plot_most_selected(sel_feat_path, save_path):
    # Function to represent the most selected features using a heatmap
    # Input:
    # - sel_feat_path: absolute path to the folder containing the selected features
    # - save_path: absolute path where to save the image

    # Ordered list of all the extracted features
    feats = ['Spectral energy', 'Relative delta power band', 'Relative theta power band',
             'Relative alpha power band', 'Relative sigma power band', 'Relative beta power band',
             'Relative gamma power band', 'delta-theta power ratio', 'delta-sigma power ratio',
             'delta-beta power ratio', 'theta-alpha power ratio', 'delta-alpha power ratio',
             'alpha-beta power ratio', 'delta-(alpha-beta) power ratio', 'theta-(alpha-beta) power ratio',
             'delta-(alpha-beta-theta) power ratio', 'Spectral centroid', 'Spectral crest factor',
             'Spectral flatness', 'Spectral skewness', 'Spectral kurtosis', 'Spectral mean', 'Spectral variance',
             'Spectral rolloff', 'Spectral spread', 'Standard deviation', 'Inter-quartile range', 'Skewness',
             'Kurtosis', 'Number of zero-crossings', 'Maximum first derivative', 'Hjorth activity',
             'Hjorth mobility', 'Hjorth complexity', 'Spectral entropy', 'Renyi entropy', 'Approximate entropy',
             'Sample entropy', 'Singular value decomposition entropy', 'Permutation entropy',
             'De-trended fluctuation analysis', 'Katz fractal dimension', 'Higuchi fractal dimension',
             'Petrosian fractal dimension', 'Lempel–Ziv complexity']

    feats_labels = ['Spectral energy', 'Relative \u03B4 power band', 'Relative \u03B8 power band',
                    'Relative \u03B1 power band', 'Relative \u03C3 power band', 'Relative \u03B2 power band',
                    'Relative \u03B3 power band', '\u03B4/\u03B8 power ratio', '\u03B4/\u03C3 power ratio',
                    '\u03B4/\u03B2 power ratio', '\u03B8/\u03B1 power ratio', '\u03B4/\u03B1 power ratio',
                    '\u03B1/\u03B2 power ratio', '\u03B4/(\u03B1 + \u03B2) power ratio',
                    '\u03B8/(\u03B1 + \u03B2) power ratio', '\u03B4/(\u03B1 + \u03B2 + \u03B8) power ratio',
                    'Spectral centroid', 'Spectral crest factor', 'Spectral flatness', 'Spectral skewness',
                    'Spectral kurtosis', 'Spectral mean', 'Spectral variance', 'Spectral rolloff', 'Spectral spread',
                    'Standard deviation', 'Inter-quartile range', 'Skewness', 'Kurtosis', 'Zero-crossings',
                    'Max first derivative', 'Hjorth activity', 'Hjorth mobility', 'Hjorth complexity',
                    'Spectral entropy', 'Renyi entropy', 'Approximate entropy', 'Sample entropy', 'SVD entropy',
                    'Permutation entropy', 'DFA exponent', 'Katz FD', 'Higuchi FD', 'Petrosian FD',
                    'Lempel–Ziv complexity']

    classes = os.listdir(sel_feat_path)

    # Matrix containing the most selected features
    matrix = np.zeros([len(feats), len(classes)])

    # Cycle on classes
    for nc, c in enumerate(classes):
        # Cycle on selected feature files
        for file in glob.glob(os.path.join(sel_feat_path, c) + '/*.npy'):
            sel_feats = np.load(file, allow_pickle=True).squeeze().tolist()
            pos_sel_feats = [feats.index(sel_feat) for sel_feat in sel_feats]
            matrix[pos_sel_feats, nc] += 1

    fig, axes = plt.subplot_mosaic([[0, '.', 1]], figsize=(10, 15), width_ratios=np.array([.7, .02, .05]))
    df = pd.DataFrame(matrix, index=feats_labels, columns=classes)
    heat_map = sb.heatmap(df, cmap='Reds', ax=axes[0], cbar=False, linecolor="black", linewidths=1,
                          xticklabels=True, square=False)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), fontsize=16)
    heat_map.set_yticklabels(heat_map.get_yticklabels(), fontsize=16)
    for _, spine in heat_map.spines.items():
        spine.set_visible(True)
    plt.tick_params(axis='both', which='major', labelsize=20, labelbottom=False, bottom=False, top=False, labeltop=True)
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=21), cmap='Reds'),
                        cax=axes[1], orientation='vertical')
    axes[1].annotate(text='Selection frequency', size=24, xy=(3, 0.5), xycoords=axes[1].transAxes, rotation=-90,
                     va="center", ha="center")
    axes[1].yaxis.set_ticks_position('left')
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'Most_selected.jpg'), dpi=300)
    # plt.show(block=False)
    # plt.pause(5)
    plt.close()


# ---------------------------------------------  JSD-FSI score definition  ---------------------------------------------

def jsd_fsi_scores(feat_path, sel_feat_path, inter_path, save_path, flag):
    # Function to define the JSD-FSI similarity-scores
    # Input:
    # - feat_path: absolute path to the feature files
    # - sel_feat_path: absolute path to the selected feature files
    # - inter_path: absolute path to the folder containing the intersection between PSG and in-ear-EEG consensus
    # - save_path: absolute path where to save the results
    # Output:
    # - jsd_fsi_path_flag: absolute path where JSD-FSI scores are saved

    # Ordered list of all the extracted features
    feats = ['Spectral energy', 'Relative delta power band', 'Relative theta power band',
             'Relative alpha power band', 'Relative sigma power band', 'Relative beta power band',
             'Relative gamma power band', 'delta-theta power ratio', 'delta-sigma power ratio',
             'delta-beta power ratio', 'theta-alpha power ratio', 'delta-alpha power ratio',
             'alpha-beta power ratio', 'delta-(alpha-beta) power ratio', 'theta-(alpha-beta) power ratio',
             'delta-(alpha-beta-theta) power ratio', 'Spectral centroid', 'Spectral crest factor',
             'Spectral flatness', 'Spectral skewness', 'Spectral kurtosis', 'Spectral mean', 'Spectral variance',
             'Spectral rolloff', 'Spectral spread', 'Standard deviation', 'Inter-quartile range', 'Skewness',
             'Kurtosis', 'Number of zero-crossings', 'Maximum first derivative', 'Hjorth activity',
             'Hjorth mobility', 'Hjorth complexity', 'Spectral entropy', 'Renyi entropy', 'Approximate entropy',
             'Sample entropy', 'Singular value decomposition entropy', 'Permutation entropy',
             'De-trended fluctuation analysis', 'Katz fractal dimension', 'Higuchi fractal dimension',
             'Petrosian fractal dimension', 'Lempel–Ziv complexity']

    classes = os.listdir(sel_feat_path)
    subs = os.listdir(feat_path)

    jsd_fsi_path = os.path.join(save_path, 'JSD-FSI scores')
    if not os.path.exists(jsd_fsi_path):
        os.makedirs(jsd_fsi_path)
    jsd_fsi_path_flag = os.path.join(jsd_fsi_path, flag)
    if not os.path.exists(jsd_fsi_path_flag):
        os.makedirs(jsd_fsi_path_flag)

    # ---------------------------------------  PSG-to-In-ear-EEG comparisons  --------------------------------------
    if flag == 'ear_vs_psg':

        chs_idx = [os.path.basename(path).split('.')[0]
                   for path in glob.glob(os.path.join(feat_path, subs[0]) + '/*.npy') if 'GDK' not in path]
        m2_ind = chs_idx.index('EEG_uni_M2')
        np.save(os.path.join(jsd_fsi_path_flag, 'M2_index.npy'), m2_ind)

        n_comps = len(os.listdir(os.path.join(feat_path, subs[0]))) - 1
        # Cycle on classes
        for nc in range(len(classes)):

            # Matrix containing final JDS-FSI scores for in-ear-EEG vs PSG
            matrix = np.zeros([n_comps, len(subs)])
            n_sel_feats = []
            comp_names = []

            # Cycle on subjects
            for n_sub, sub in enumerate(subs):

                # Not considering subjects 3 and 8 for REM stage
                if (nc == 2) & (n_sub in [2, 7]): flag_1 = 0
                else: flag_1 = 1

                if flag_1 == 1:
                    # Loading reference
                    ref_file = np.load(os.path.join(inter_path, sub + '.npy')).squeeze()
                    # Extracting the indexes for Awake (0), NREM (1), and REM (2) stages
                    ref = np.argwhere(ref_file == nc).squeeze()

                    # Loading in-ear-EEG feature file
                    gdk_feats = np.load(glob.glob(os.path.join(feat_path, sub) + '/GDK*.npy')[0]).squeeze()

                    # Loading paths to PSG feature files
                    psg_chs = [path for path in glob.glob(os.path.join(feat_path, sub) + '/*.npy') if 'GDK' not in path]

                    n_comp = 0
                    for psg_ch in psg_chs:
                        # Not considering channel M2 for subjects 3 and 6
                        if (n_sub in [2, 5]) & (n_comp == m2_ind): n_comp += 1

                        comp = 'GDK_' + os.path.basename(psg_ch).split('.')[0].split('_')[-1]
                        sel_feats = np.load(os.path.join(sel_feat_path, classes[nc], comp + '.npy'),
                                            allow_pickle=True).squeeze().tolist()

                        if n_sub == 0:
                            n_sel_feats.append(len(sel_feats))
                            if nc == 0:
                                comp_names.append(os.path.basename(psg_ch).split('.')[0].split('_')[-1])

                        # Cycle on selected features
                        for feat in sel_feats:
                            if len(ref) > 0:
                                psg_feat = np.load(psg_ch).squeeze()[feats.index(feat), ref]
                                gdk_feat = gdk_feats[feats.index(feat), ref]

                                # Evaluating probability distributions using kernel density estimation (KDE)
                                kde_psg = gaussian_kde(psg_feat)
                                kde_gdk = gaussian_kde(gdk_feat)
                                common_sup = np.linspace(np.min([np.min(gdk_feat), np.min(psg_feat)]),
                                                         np.max([np.max(gdk_feat), np.max(psg_feat)]), 1000)
                                pd_psg = kde_psg(common_sup)
                                pd_gdk = kde_gdk(common_sup)

                                js = jensenshannon(pd_gdk, pd_psg)
                                if math.isinf(js): js = 1
                                matrix[n_comp, n_sub] += 1 - (js ** 2)
                        n_comp += 1

                if (nc == 0) & (n_sub == 0):
                    np.save(os.path.join(jsd_fsi_path_flag, 'Comparisons_ear_psg.npy'), comp_names)

            # Normalizing by the number of selected features
            for i in range(len(n_sel_feats)):
                matrix[i, :] /= n_sel_feats[i]
            np.save(os.path.join(jsd_fsi_path_flag, 'JSD_FSI_' + classes[nc] + '.npy'), matrix)

    # ------------------------------------------  PSG-to-PSG comparisons  ------------------------------------------
    elif flag == 'psg_vs_psg':

        m2_count = 0
        m2_ind = []
        m2_names = []
        psg_chs = [os.path.basename(path).split('.')[0].split('_')[-1]
                   for path in glob.glob(os.path.join(feat_path, subs[0]) + '/*.npy') if 'GDK' not in path]
        for n_comp_1 in range(len(psg_chs) - 1):
            for n_comp_2 in np.arange(n_comp_1 + 1, len(psg_chs)):
                if (psg_chs[n_comp_1] == 'M2') | (psg_chs[n_comp_2] == 'M2'):
                    m2_ind.append(m2_count)
                    m2_names.append(psg_chs[n_comp_1] + '_' + psg_chs[n_comp_2])
                m2_count += 1
        np.save(os.path.join(jsd_fsi_path_flag, 'M2_index.npy'), m2_ind)

        n = len(os.listdir(os.path.join(feat_path, subs[0]))) - 1
        n_comps = int((n * (n - 1)) / 2)
        # Cycle on classes
        for nc in range(len(classes)):

            # Matrix containing final JDS-FSI scores for PSG vs PSG
            matrix = np.zeros([n_comps, len(subs)])
            n_sel_feats = []
            comp_names = []

            # Cycle on subjects
            for n_sub, sub in enumerate(subs):

                # Not considering subjects 3 and 8 for REM stage
                if (nc == 2) & (n_sub in [2, 7]): flag_1 = 0
                else: flag_1 = 1

                if flag_1 == 1:
                    # Loading reference
                    ref_file = np.load(os.path.join(inter_path, sub + '.npy')).squeeze()
                    # Extracting the indexes for Awake (0), NREM (1), and REM (2) stages
                    ref = np.argwhere(ref_file == nc).squeeze()

                    # Loading paths to PSG feature files
                    psg_chs = [path for path in glob.glob(os.path.join(feat_path, sub) + '/*.npy') if 'GDK' not in path]

                    count = 0
                    for n_comp_1 in range(len(psg_chs) - 1):
                        for n_comp_2 in np.arange(n_comp_1 + 1, len(psg_chs)):

                            # Not considering channel M2 for subjects 3 and 6
                            if n_sub in [2, 5]:
                                while count in m2_ind:
                                    count += 1

                            comp = (os.path.basename(psg_chs[n_comp_1]).split('.')[0].split('_')[-1] + '_' +
                                    os.path.basename(psg_chs[n_comp_2]).split('.')[0].split('_')[-1])
                            sel_feats = np.load(os.path.join(sel_feat_path, classes[nc], comp + '.npy'),
                                                allow_pickle=True).squeeze().tolist()

                            if n_sub == 0:
                                n_sel_feats.append(len(sel_feats))
                                if nc == 0:
                                    comp_names.append(comp)

                            # Cycle on selected features
                            for feat in sel_feats:
                                if len(ref) > 1:
                                    psg_feat_1 = np.load(psg_chs[n_comp_1]).squeeze()[feats.index(feat), ref]
                                    psg_feat_2 = np.load(psg_chs[n_comp_2]).squeeze()[feats.index(feat), ref]

                                    # Evaluating probability distributions using kernel density estimation (KDE)
                                    kde_psg_1 = gaussian_kde(psg_feat_1)
                                    kde_psg_2 = gaussian_kde(psg_feat_2)
                                    common_sup = np.linspace(np.min([np.min(psg_feat_1), np.min(psg_feat_2)]),
                                                             np.max([np.max(psg_feat_1), np.max(psg_feat_2)]), 1000)
                                    pd_psg_1 = kde_psg_1(common_sup)
                                    pd_psg_2 = kde_psg_2(common_sup)

                                    js = jensenshannon(pd_psg_1, pd_psg_2)
                                    if math.isinf(js): js = 1
                                    matrix[count, n_sub] += 1 - (js ** 2)
                            count += 1

                    if (nc == 0) & (n_sub == 0):
                        np.save(os.path.join(jsd_fsi_path_flag, 'Comparisons_psg_psg.npy'), comp_names)

            # Normalizing by the number of selected features
            for i in range(len(n_sel_feats)):
                matrix[i, :] /= n_sel_feats[i]
            np.save(os.path.join(jsd_fsi_path_flag, 'JSD_FSI_' + classes[nc] + '.npy'), matrix)

    return jsd_fsi_path_flag


def create_head():
    # Function to create a model of the head
    # Output:
    # - RawMFF file containing the model

    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = ['F4']
    data = np.zeros((len(ch_names), 1))
    info = mne.create_info(ch_names, sfreq=1000, ch_types='eeg')
    raw = mne.io.RawArray(data=data, info=info)
    raw.set_montage(montage)

    return raw


def head_plot(jsd_fsi_path, save_path):
    # Function to plot PSG derivations on head
    # Input:
    # - jsd_fsi_path: absolute path to the folder containing JSD-FSI scores
    # - save_path: absolute path where to save the image

    classes = [os.path.basename(file).split('.')[0].split('_')[-1]
               for file in glob.glob(jsd_fsi_path + '/JSD_FSI_*.npy')]

    # Loading the channels by the order they are analysed
    comp_ord = np.load(glob.glob(jsd_fsi_path + '/Comparisons*.npy')[0], allow_pickle=True).squeeze().tolist()

    # ----------------------------------------------------------------------------------------------------------- #
    # (x, y) coordinates for all pairs of EEG and EOG bipolar channels i.e., C3M2 (M2, C3), C4M1 (M1, C4),
    # F3M2 (M2, F3), F4M1 (M1, F4), O1M2 (M2, O1), O2M1 (M1, O2), E1M1 (M1, E1), E1M2 (M2, E1), E2M1 (M1, E2),
    # E2M2 (M2, E2), M2M1 (M1, M2)
    bi_ord = ['C3M2', 'C4M1', 'F3M2', 'F4M1', 'O1M2', 'O2M1', 'E1M1', 'E1M2', 'E2M1', 'E2M2', 'M2M1']
    bi_pos = [comp_ord.index(bi_ord_ch) for bi_ord_ch in bi_ord]
    eeg_eog_bi_x_couples = np.array([[0.0998, -0.0426], [-0.1023, 0.0388], [0.1008, -0.0355], [-0.1038, 0.0329],
                                     [0.0988, -0.0221], [-0.1007, 0.01999], [-0.1048, -0.0658], [0.112, -0.061],
                                     [-0.1038, 0.0613], [0.1028, 0.063], [-0.1033, 0.0986]])
    eeg_eog_bi_y_couples = np.array([[-0.0215, 0.018], [-0.0205, 0.0175], [-0.021, 0.0646], [-0.0205, 0.0641],
                                     [-0.0255, -0.0576], [-0.0257, -0.059], [-0.0185, 0.1005], [-0.0275, 0.0975],
                                     [-0.0195, 0.1005], [-0.021, 0.1005], [-0.0225, -0.0235]])
    # (x, y) coordinates for all pairs of EEG and EOG unipolar channels i.e., 'C3', 'C4', 'F3', 'F4', 'O1', 'O2',
    # 'E1', 'E2', 'M1', 'M2'
    eeg_eog_uni_x_couples = np.array([-0.0461, 0.046, -0.0385, 0.036, -0.0253,
                                      0.0208, -0.0669, 0.0613, -0.1074, 0.1028])
    eeg_eog_uni_y_couples = np.array([0.019, 0.0195, 0.0646, 0.0662, -0.06,
                                      -0.0598, 0.099, 0.1005, -0.0225, -0.024])
    # text labels and corresponding (x, y) coordinates
    uni_ord = ['C3', 'C4', 'F3', 'F4', 'O1', 'O2', 'E1', 'E2', 'M1', 'M2']
    uni_pos = [comp_ord.index(uni_ord_ch) for uni_ord_ch in uni_ord]
    eeg_eog_uni_x_text = np.array([-0.083, 0.06, -0.073, 0.049, -0.065, 0.035, -0.102, 0.078, -0.128, 0.095])
    eeg_eog_uni_y_text = np.array([0.023, 0.022, 0.065, 0.0633, -0.063, -0.0635, 0.097, 0.098, -0.052, -0.052])
    # ----------------------------------------------------------------------------------------------------------- #

    # Finding the minimum JSD-FSI score among subjects and classes
    # JSD-FSI ranges between 0 and 1
    abs_min = 2
    avg_sub = {}
    std_sub = {}
    jsd_fsi_bi = {}
    jsd_fsi_uni = {}
    for nc, c in enumerate(classes):
        # Matrix dimensions are [Number of comparisons X Number of subjects]
        # The order of comparisons i.e., PSG channels is given by comp_ord
        # however, it should be ['C3M2', 'C4M1', 'F3M2', 'F4M1', 'M2M1', 'O1M2', 'O2M1', 'C3', 'C4', 'F3', 'F4', 'M1',
        # 'M2', 'O1', 'O2', 'E1M1', 'E1M2', 'E2M1', 'E2M2', 'E1', 'E2']
        matrix = np.load(glob.glob(jsd_fsi_path + '/JSD_FSI_' + c + '.npy')[0]).squeeze()

        # Note:
        # 1. The channel M2 for subjects 3 and 6 must be excluded, thus to avoid bias while computing the absolute
        # minimum the corresponding values are set equal to 2, as the JSD-FSI ranges between 0 and 1
        # 2. For REM stage, subjects 3 and 8 must be excluded as well for the evaluation of the absolute minimum

        if nc == 0: sbj_names = ['Subject_' + f"{(n + 1):02d}" for n in range(np.size(matrix, 1))]

        avg_sub[nc] = np.zeros(len(sbj_names))
        std_sub[nc] = np.zeros(len(sbj_names))
        for ns in range(len(sbj_names)):
            if (nc == 2) & (ns in [2, 7]): flag = 0
            else: flag = 1

            if flag == 1:
                if ns in [2, 5]: m = np.delete(matrix[:, ns], comp_ord.index('M2'))
                else: m = matrix[:, ns].copy()
                # For each subject and class, finding average and standard deviation values among all channels
                avg_sub[nc][ns] = np.round(np.mean(m), 2)
                std_sub[nc][ns] = np.round(np.std(m), 2)
                # Finding the absolute minimum among all subjects, channels and classes
                if np.min(m) < abs_min: abs_min = np.round(np.min(m), 1)

        # Rearranging the channels
        jsd_fsi_bi[nc] = matrix[bi_pos, :]
        jsd_fsi_uni[nc] = matrix[uni_pos, :]

    for nc, c in enumerate(classes):

        head = create_head()
        fig, axes = plt.subplot_mosaic([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 10, 10, 10, 10]],
                                       layout='constrained', figsize=(14, 7),
                                       height_ratios=np.array([.4925, .4925, .015]))

        for ns in range(len(sbj_names)):

            head.plot_sensors(show_names=False, show=False, sphere=(-0.002, 0.009, 0, 0.108), axes=axes[ns])

            # 1. EEG and EOG bipolar channels
            for n in range(np.size(eeg_eog_bi_x_couples, 0)):
                segments = np.column_stack([eeg_eog_bi_x_couples[n, :], eeg_eog_bi_y_couples[n, :]])
                segments = np.array([segments])

                # Subjects 3 and 8 do not have any epochs in REM stage
                if (nc == 2) & ((ns == 2) | (ns == 7)):
                    lc = LineCollection(segments, array=np.array([.5, .5]), cmap=colormaps['binary'],
                                        linestyles='--', linewidth=2, norm=Normalize(0, 1))
                else:
                    lc = LineCollection(segments, array=np.array([jsd_fsi_bi[nc][n, ns], jsd_fsi_bi[nc][n, ns]]),
                                        cmap=colormaps['plasma'], linewidth=2, norm=Normalize(vmin=0, vmax=1))
                axes[ns].add_collection(lc)

            # 2. EEG and EOG unipolar channels
            for n in range(np.size(eeg_eog_uni_x_couples)):
                if ((nc == 2) & ((ns == 2) | (ns == 7))) | (((ns == 2) | (ns == 5)) & (uni_ord[n] == 'M2')):
                    axes[ns].scatter(eeg_eog_uni_x_couples[n], eeg_eog_uni_y_couples[n], c=0, cmap='binary', marker='o',
                                     s=250, edgecolor='k', vmin=0, vmax=1, zorder=2)
                else:
                    axes[ns].scatter(eeg_eog_uni_x_couples[n], eeg_eog_uni_y_couples[n], c=jsd_fsi_uni[nc][n, ns],
                                     cmap='plasma', marker='o', s=250, edgecolor='k', vmin=0, vmax=1, zorder=2)
                axes[ns].text(eeg_eog_uni_x_text[n], eeg_eog_uni_y_text[n], uni_ord[n], fontsize=13)

            if (nc == 2) & ((ns == 2) | (ns == 7)):
                axes[ns].set_title(sbj_names[ns].replace('_', ' ') + ' (None)')
            else:
                axes[ns].set_title(sbj_names[ns].replace('_', ' ') + ' (' + str(avg_sub[nc][ns]) + ' '
                                   + u"\u00B1" + ' ' + str(std_sub[nc][ns]) + ')', fontsize=12)

        fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap='plasma'),
                     cax=axes[10], orientation='horizontal')
        axes[10].annotate(text='JSD-FSI Similarity-score for ' + c + ' stage', size=18, xy=(.5, 1.5),
                          xycoords=axes[10].transAxes, va="bottom", ha="center")
        axes[10].tick_params(labelsize=12)
        fig.savefig(os.path.join(save_path, 'Similarity_' + c + '_Head_Plot.jpg'), dpi=300)
        # plt.show(block=False)
        # plt.pause(8)
        plt.close()


def hist_plot(jsd_fsi_path_ear, jsd_fsi_path_psg, save_path):
    # Function to represent the histograms for PSG-to-In-ear-EEG and PSG-to-PSG comparisons
    # Input:
    # - jsd_fsi_path_ear: absolute path to the folder containing JSD-FSI scores for the PSG-to-In-ear-EEG comparison
    # - jsd_fsi_path_psg: absolute path to the folder containing JSD-FSI scores for the PSG-to-PSG comparison
    # - save_path: absolute path where to save the image

    m2_ind_ear = np.load(os.path.join(jsd_fsi_path_ear, 'M2_index.npy')).squeeze().tolist()
    m2_ind_psg = np.load(os.path.join(jsd_fsi_path_psg, 'M2_index.npy')).squeeze().tolist()

    # Parameters for the plot
    x_low_lim = [0.57, 0.38, 0.35]
    y_sup_lim = [55, 86, 46]
    x_ticks_c = [np.arange(0.6, 1.01, 0.1), np.arange(0.4, 1.01, 0.2), np.arange(0.4, 1.01, 0.2)]
    y_ticks_c = [np.arange(0, 60, 10), np.arange(0, 100, 20), np.arange(0, 50, 10)]

    classes = [os.path.basename(file).split('.')[0].split('_')[-1]
               for file in glob.glob(jsd_fsi_path_ear + '/JSD_FSI_*.npy')]
    for nc, c in enumerate(classes):
        jsd_fsi_ear = np.load(glob.glob(jsd_fsi_path_ear + '/JSD_FSI_' + c + '.npy')[0]).squeeze()
        jsd_fsi_psg = np.load(glob.glob(jsd_fsi_path_psg + '/JSD_FSI_' + c + '.npy')[0]).squeeze()

        if nc == 0: sbj_names = ['Subject_' + f"{(n + 1):02d}" for n in range(np.size(jsd_fsi_ear, 1))]

        # As the JSD-FSI ranges between [0, 1] --> x_min and x_max are initialized out of this range
        x_min = 2
        x_max = -1
        for ns in range(len(sbj_names)):
            if (nc == 2) & (ns in [2, 7]): min_max = 0
            else: min_max = 1

            if min_max == 1:
                if ns in [2, 5]:
                    scores_ear = np.delete(jsd_fsi_ear[:, ns], m2_ind_ear)
                    scores_psg = np.delete(jsd_fsi_psg[:, ns], m2_ind_psg)
                else:
                    scores_ear = jsd_fsi_ear[:, ns].copy()
                    scores_psg = jsd_fsi_psg[:, ns].copy()

                if np.min([np.min(scores_ear), np.min(scores_psg)]) < x_min:
                    x_min = np.min([np.min(scores_ear), np.min(scores_psg)])
                if np.max([np.max(scores_ear), np.max(scores_psg)]) > x_max:
                    x_max = np.max([np.max(scores_ear), np.max(scores_psg)])

        fig, axes = plt.subplots(figsize=(14, 10), nrows=2, ncols=int(np.size(jsd_fsi_ear, 1) / 2))
        # Cycle on subjects
        for ns in range(np.size(jsd_fsi_ear, 1)):
            # Subjects 3 and 8 do not have any epochs in REM stage
            if (nc == 2) & ((ns == 2) | (ns == 7)): flag = 0
            else: flag = 1

            if ns < 5: row = 0; col = ns
            else: row = 1; col = ns - 5

            if flag == 1:

                if ns in [2, 5]:
                    scores_psg = np.delete(jsd_fsi_psg[:, ns].copy(), m2_ind_psg)
                    scores_gdk = np.delete(jsd_fsi_ear[:, ns].copy(), m2_ind_ear)

                else:
                    scores_psg = jsd_fsi_psg[:, ns].copy()
                    scores_gdk = jsd_fsi_ear[:, ns].copy()

                axes[row, col].hist(scores_psg, bins=20, edgecolor='black', label='PSG-to-PSG',
                                    facecolor=np.array([241 / 255, 126 / 255, 125 / 255, 100 / 255]),
                                    range=(x_min, x_max))
                axes[row, col].hist(scores_gdk, bins=20, edgecolor='black', label='PSG-to-In-ear-EEG',
                                    facecolor=np.array([98 / 255, 149 / 255, 235 / 255, 100 / 255]),
                                    range=(x_min, x_max))

                axes[row, col].spines[['right', 'top']].set_visible(False)
                axes[row, col].set_xlim([x_low_lim[nc], 1.01])
                axes[row, col].set_ylim([0, y_sup_lim[nc]])
                axes[row, col].set_xticks(x_ticks_c[nc])
                axes[row, col].set_yticks(y_ticks_c[nc])
                axes[row, col].set_title(sbj_names[ns].replace('_', ' '), fontsize=12)

                if (row == 0) & (col == 0):
                    axes[row, col].legend(bbox_to_anchor=(0., 1.1, 6.2, .102), loc='lower left', ncols=2,
                                          mode="expand", borderaxespad=0., fontsize=11)

            else:
                axes[row, col].plot(np.array([0, 1]), np.array([0, 1]), 'k')
                axes[row, col].plot(np.array([0, 1]), np.array([1, 0]), 'k')
                axes[row, col].set_xlim([-.5, 1.5])
                axes[row, col].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                axes[row, col].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                axes[row, col].set_title(sbj_names[ns].replace('_', ' '), fontsize=12)

            fig.suptitle(c + ' stage', fontsize=16, y=.99)
            fig.supxlabel('JSD-FSI Similarity-scores', fontsize=14)
            fig.supylabel('# Occurrences', fontsize=14, x=0.015)
            fig.tight_layout(h_pad=3.5)
            fig.savefig(os.path.join(save_path, 'Hist_' + c + '.jpg'), dpi=300)
            # fig.show(block=False)
            # fig.pause(8)
            plt.close()


def separate_hist_plot(jsd_fsi_path_ear, jsd_fsi_path_psg, save_path):
    # Function to represent the histograms for PSG-to-In-ear-EEG and PSG-to-PSG comparisons
    # Input:
    # - jsd_fsi_path_ear: absolute path to the folder containing JSD-FSI scores for the PSG-to-In-ear-EEG comparison
    # - jsd_fsi_path_psg: absolute path to the folder containing JSD-FSI scores for the PSG-to-PSG comparison
    # - save_path: absolute path where to save the image

    classes = [os.path.basename(file).split('.')[0].split('_')[-1]
               for file in glob.glob(jsd_fsi_path_ear + '/JSD_FSI_*.npy')]

    eog_chs = ['E1M1', 'E1M2', 'E2M1', 'E2M2', 'E1', 'E2']
    eeg_chs = ['C3M2', 'C4M1', 'F3M2', 'F4M1', 'M2M1', 'O1M2', 'O2M1', 'C3', 'C4', 'F3', 'F4', 'M1', 'M2', 'O1', 'O2']
    ear_comps = np.load(glob.glob(jsd_fsi_path_ear + '/Comparisons*.npy')[0], allow_pickle=True).squeeze().tolist()
    ear_eog_pos = [ear_comps.index(eog_ch) for eog_ch in eog_chs]
    ear_eeg_pos = [ear_comps.index(eeg_ch) for eeg_ch in eeg_chs]
    m2_ear = ear_comps.index('M2')

    eog_eog_chs = []
    eog_eeg_chs = []
    eeg_eeg_chs = []
    m2_eog = []
    m2_eeg = []
    psg_comps = np.load(glob.glob(jsd_fsi_path_psg + '/Comparisons*.npy')[0], allow_pickle=True).squeeze().tolist()
    for psg_id, psg_comp in enumerate(psg_comps):
        psg_1, psg_2 = psg_comp.split('_')
        if psg_1 in eog_chs:
            if psg_2 in eog_chs: eog_eog_chs.append(psg_id)
            if psg_2 in eeg_chs:
                eog_eeg_chs.append(psg_id)
                if psg_2 == 'M2': m2_eog.append(eog_eeg_chs.index(psg_id))
        if psg_1 in eeg_chs:
            if psg_2 in eog_chs:
                eog_eeg_chs.append(psg_id)
                if psg_1 == 'M2': m2_eog.append(eog_eeg_chs.index(psg_id))
            if psg_2 in eeg_chs:
                eeg_eeg_chs.append(psg_id)
                if (psg_1 == 'M2') | (psg_2 == 'M2'): m2_eeg.append(eeg_eeg_chs.index(psg_id))

    # Parameters for the plot
    x_low_lim = [0.57, 0.38, 0.35]
    y_sup_lim_eeg = [40, 46, 30]
    y_sup_lim_eog = [11.5, 8.5, 10.5]
    x_ticks_c = [np.arange(0.6, 1.01, 0.1), np.arange(0.4, 1.01, 0.2), np.arange(0.4, 1.01, 0.2)]
    y_ticks_c_eeg = [np.arange(0, 50, 10), np.arange(0, 50, 10), np.arange(0, 40, 10)]
    y_ticks_c_eog = [np.arange(0, 12, 2), np.arange(0, 10, 2), np.arange(0, 12, 2)]
    ext_eeg = [5.9, 5.935, 5.94]
    ext_eog = [5.9, 5.75, 5.94]
    color_line = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
    color_hist = color_line.copy()
    for i in range(len(color_hist)):
        color_hist[i][-1] = 0.5
    smooth_points = 100

    m2_ind_ear = np.load(os.path.join(jsd_fsi_path_ear, 'M2_index.npy')).squeeze().tolist()
    m2_ind_psg = np.load(os.path.join(jsd_fsi_path_psg, 'M2_index.npy')).squeeze().tolist()

    for nc, c in enumerate(classes):
        # Rearranging JSD-FSI scores for PSG-to-In-ear-EEG comparisons
        matrix_ear = np.load(glob.glob(jsd_fsi_path_ear + '/JSD_FSI_' + c + '.npy')[0]).squeeze()
        ear_eog = matrix_ear[ear_eog_pos, :]
        ear_eeg = matrix_ear[ear_eeg_pos, :]

        # Rearranging JSD-FSI scores for PSG-to-PSG comparisons
        matrix_psg = np.load(glob.glob(jsd_fsi_path_psg + '/JSD_FSI_' + c + '.npy')[0]).squeeze()
        eog_eog = matrix_psg[eog_eog_chs, :]
        eog_eeg = matrix_psg[eog_eeg_chs, :]
        eeg_eeg = matrix_psg[eeg_eeg_chs, :]

        # As the JSD-FSI ranges between [0, 1] --> x_min and x_max are initialized out of this range
        x_min = 2
        x_max = -1
        for ns in range(np.size(matrix_ear, 1)):
            if (nc == 2) & (ns in [2, 7]): min_max = 0
            else: min_max = 1

            if min_max == 1:
                if ns in [2, 5]:
                    scores_ear = np.delete(matrix_ear[:, ns], m2_ind_ear)
                    scores_psg = np.delete(matrix_psg[:, ns], m2_ind_psg)
                else:
                    scores_ear = matrix_ear[:, ns].copy()
                    scores_psg = matrix_psg[:, ns].copy()

                if np.min([np.min(scores_ear), np.min(scores_psg)]) < x_min:
                    x_min = np.min([np.min(scores_ear), np.min(scores_psg)])
                if np.max([np.max(scores_ear), np.max(scores_psg)]) > x_max:
                    x_max = np.max([np.max(scores_ear), np.max(scores_psg)])

        # Figure Scalp-EEG-to-In-ear-EEG vs Scalp-EEG-to-Scalp-EEG
        fig_1, axes_1 = plt.subplots(figsize=(14, 10), nrows=2, ncols=int(np.size(matrix_ear, 1) / 2))
        # Figure EOG-to-In-ear-EEG vs EOG-to-EOG
        fig_2, axes_2 = plt.subplots(figsize=(14, 10), nrows=2, ncols=int(np.size(matrix_ear, 1) / 2))

        for ns in range(np.size(matrix_ear, 1)):

            if (nc == 2) & ((ns == 2) | (ns == 7)): flag = 0
            else: flag = 1

            if ns < 5: row = 0; col = ns
            else: row = 1; col = ns - 5

            if flag == 1:
                if ns in [2, 5]:
                    eeg_eeg_scores = np.delete(eeg_eeg[:, ns], m2_eeg)
                    ear_eeg_scores = np.delete(ear_eeg[:, ns], m2_ear)
                    eog_eog_scores = eog_eog[:, ns].copy()
                    ear_eog_scores = ear_eog[:, ns].copy()
                    eog_eeg_scores = np.delete(eog_eeg[:, ns], m2_eog)
                else:
                    eeg_eeg_scores = eeg_eeg[:, ns].copy()
                    ear_eeg_scores = ear_eeg[:, ns].copy()
                    eog_eog_scores = eog_eog[:, ns].copy()
                    ear_eog_scores = ear_eog[:, ns].copy()
                    eog_eeg_scores = eog_eeg[:, ns].copy()

                hist1, bin_edges1 = np.histogram(eeg_eeg_scores, bins=20, range=(x_min, x_max))
                bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
                smooth_bin_centers1 = np.linspace(bin_centers1.min(), bin_centers1.max(), smooth_points)
                f1 = interp1d(bin_centers1, hist1, kind='cubic')
                smooth_hist1 = f1(smooth_bin_centers1)
                smooth_hist1[(smooth_bin_centers1 < np.min(eeg_eeg_scores)) |
                             (smooth_bin_centers1 > np.max(eeg_eeg_scores))] = 0
                axes_1[row, col].plot(smooth_bin_centers1, smooth_hist1, label='Scalp-EEG-to-Scalp-EEG',
                                      color=color_line[0], linewidth=1.5)
                # axes_1[row, col].hist(eeg_eeg_scores, bins=20, edgecolor='black',
                #                       facecolor=color_hist[0], range=(x_min, x_max))

                hist2, bin_edges2 = np.histogram(ear_eeg_scores, bins=20, range=(x_min, x_max))
                bin_centers2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2
                smooth_bin_centers2 = np.linspace(bin_centers2.min(), bin_centers2.max(), smooth_points)
                f2 = interp1d(bin_centers2, hist2, kind='cubic')
                smooth_hist2 = f2(smooth_bin_centers2)
                smooth_hist2[(smooth_bin_centers2 < np.min(ear_eeg_scores)) |
                             (smooth_bin_centers2 > np.max(ear_eeg_scores))] = 0
                axes_1[row, col].plot(smooth_bin_centers2, smooth_hist2, label='Scalp-EEG-to-In-ear-EEG',
                                      color=color_line[1], linewidth=1.5)
                # axes_1[row, col].hist(ear_eeg_scores, bins=20, edgecolor='black',
                #                       facecolor=color_hist[1], range=(x_min, x_max))

                axes_1[row, col].fill_between(smooth_bin_centers2, smooth_hist2,
                                              where=(smooth_bin_centers2 > np.min(eeg_eeg_scores)), color=color_hist[4])

                hist3, bin_edges3 = np.histogram(eog_eog_scores, bins=20, range=(x_min, x_max))
                bin_centers3 = (bin_edges3[:-1] + bin_edges3[1:]) / 2
                smooth_bin_centers3 = np.linspace(bin_centers3.min(), bin_centers3.max(), smooth_points)
                f3 = interp1d(bin_centers3, hist3, kind='cubic')
                smooth_hist3 = f3(smooth_bin_centers3)
                smooth_hist3[(smooth_bin_centers3 < np.min(eog_eog_scores)) |
                             (smooth_bin_centers3 > np.max(eog_eog_scores))] = 0
                axes_2[row, col].plot(smooth_bin_centers3, smooth_hist3, label='EOG-to-EOG',
                                      color=color_line[2], linewidth=1.5)
                # axes_2[row, col].hist(eog_eog_scores, bins=20, edgecolor='black',
                #                       facecolor=color_hist[2], range=(x_min, x_max))

                hist4, bin_edges4 = np.histogram(ear_eog_scores, bins=20, range=(x_min, x_max))
                bin_centers4 = (bin_edges4[:-1] + bin_edges4[1:]) / 2
                smooth_bin_centers4 = np.linspace(bin_centers4.min(), bin_centers4.max(), smooth_points)
                f4 = interp1d(bin_centers4, hist4, kind='cubic')
                smooth_hist4 = f4(smooth_bin_centers4)
                smooth_hist4[(smooth_bin_centers4 < np.min(ear_eog_scores)) |
                             (smooth_bin_centers4 > np.max(ear_eog_scores))] = 0
                axes_2[row, col].plot(smooth_bin_centers4, smooth_hist4, label='EOG-to-In-ear-EEG',
                                      color=color_line[3], linewidth=1.5)
                # axes_2[row, col].hist(ear_eog_scores, bins=20, edgecolor='black',
                #                       facecolor=color_hist[3], range=(x_min, x_max))

                axes_2[row, col].fill_between(smooth_bin_centers4, smooth_hist4,
                                              where=(smooth_bin_centers4 > np.min(eog_eog_scores)), color=color_hist[4])

                axes_1[row, col].spines[['right', 'top']].set_visible(False)
                axes_1[row, col].set_xlim([x_low_lim[nc], 1.01])
                axes_1[row, col].set_ylim([0, y_sup_lim_eeg[nc]])
                axes_1[row, col].set_xticks(x_ticks_c[nc])
                axes_1[row, col].set_yticks(y_ticks_c_eeg[nc])
                axes_1[row, col].set_title('Subject ' + str(ns + 1), fontsize=12)

                axes_2[row, col].spines[['right', 'top']].set_visible(False)
                axes_2[row, col].set_xlim([x_low_lim[nc], 1.01])
                axes_2[row, col].set_ylim([0, y_sup_lim_eog[nc]])
                axes_2[row, col].set_xticks(x_ticks_c[nc])
                axes_2[row, col].set_yticks(y_ticks_c_eog[nc])
                axes_2[row, col].set_title('Subject ' + str(ns + 1), fontsize=12)

                if (row == 0) & (col == 0):
                    axes_1[row, col].legend(bbox_to_anchor=(0., 1.1, ext_eeg[nc], .102), loc='lower left', ncols=2,
                                            mode="expand", borderaxespad=0., fontsize=11)
                    axes_2[row, col].legend(bbox_to_anchor=(0., 1.1, ext_eog[nc], .102), loc='lower left', ncols=2,
                                            mode="expand", borderaxespad=0., fontsize=11)

            else:
                axes_1[row, col].plot(np.array([0, 1]), np.array([0, 1]), 'k')
                axes_1[row, col].plot(np.array([0, 1]), np.array([1, 0]), 'k')
                axes_1[row, col].set_xlim([-.5, 1.5])
                axes_1[row, col].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                axes_1[row, col].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                axes_1[row, col].set_title('Subject ' + str(ns + 1), fontsize=12)

                axes_2[row, col].plot(np.array([0, 1]), np.array([0, 1]), 'k')
                axes_2[row, col].plot(np.array([0, 1]), np.array([1, 0]), 'k')
                axes_2[row, col].set_xlim([-.5, 1.5])
                axes_2[row, col].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                axes_2[row, col].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                axes_2[row, col].set_title('Subject ' + str(ns + 1), fontsize=12)

        fig_1.suptitle(c + ' stage', fontsize=16, y=.99)
        fig_1.supxlabel('JSD-FSI Similarity-scores', fontsize=14)
        fig_1.supylabel('# Occurrences', fontsize=14, x=0.015)
        fig_1.tight_layout(h_pad=3.5)
        fig_1.savefig(os.path.join(save_path, 'Separate_Scalp_Hist_' + c + '.jpg'), dpi=300)

        fig_2.suptitle(c + ' stage', fontsize=16, y=.99)
        fig_2.supxlabel('JSD-FSI Similarity-scores', fontsize=14)
        fig_2.supylabel('# Occurrences', fontsize=14, x=0.015)
        fig_2.tight_layout(h_pad=3.5)
        fig_2.savefig(os.path.join(save_path, 'Separate_EOG_Hist_' + c + '.jpg'), dpi=300)

        # plt.show(block=False)
        # plt.pause(8)
        plt.close()


# -----------------------------------------------  Statistical analyses  -----------------------------------------------


def statistical_analysis(m, flag='two-sided', alpha=0.05):
    # Function to perform statistical analyses on final results
    # Input:
    # - m: matrix with dimensions [Number of distributions to compare X Number of samples].
    #      Alternatively, m can be a list of arrays.
    #      Note: the code has been implemented to compare two or three distributions
    # - flag: if 'two-sided', checking possible significant difference among distributions i.e., performing a two-sided
    #         test; if 'one-sided', checking whether the one distribution is lower/greater than another
    # - alpha: significance level to use for statistical tests

    assert flag in ['two-sided', 'one-sided'], "flag value not correctly defined"
    assert isinstance(m, list), "Distributions must be provided within a list"
    assert len(m) in [2, 3], "There must be at least 2 or at most 3 distributions to compare"

    # -------------------------------------------  Checking normality  ---------------------------------------------
    print('\nChecking normality')

    # Checking normality of distributions to compare according to Shapiro-Wilk test
    # If all distributions are normal --> a parametric statistical test is used; otherwise a non-parametric statistical
    # test is used. A counter is defined to check this aspect
    count_no_norm = 0
    for ind in range(len(m)):
        s_norm, p_norm = shapiro(m[ind])
        # if p < alpha --> the distribution is not normal
        if p_norm < alpha:
            print('p-value for distribution ' + str(ind) + ': ' + str(p_norm))
            count_no_norm += 1
    # Printing results
    if count_no_norm == 0: print('All distributions are normal')
    else: print('NOT all distributions are normal')

    # ---------------------------------------  Performing statistical tests  ---------------------------------------
    print('\nStatistical analysis')

    # Two distributions to compare
    if len(m) == 2:
        # Two-sided statistical test
        if flag == 'two-sided':
            # Parametric statistical test
            if count_no_norm == 0: s_1, p_1 = ttest_ind(m[0], m[1])
            # Non-parametric statistical test
            else: s_1, p_1 = mannwhitneyu(m[0], m[1])
            # Printing results
            if p_1 < alpha: print('There is statistical significant difference - ' + str(p_1))
            else: print('There is no statistical significant difference')

        # One-sided statistical test
        else:
            # Verifying first whether the first distribution is lower than the other and then if it is greater than
            # the other
            for alt in ['less', 'greater']:
                # Parametric statistical tests
                if count_no_norm == 0: s_2, p_2 = ttest_ind(m[0], m[1], alternative=alt)
                # Non-parametric statistical tests
                else: s_2, p_2 = mannwhitneyu(m[0], m[1], alternative=alt)
                # Printing results
                if p_2 < alpha: print('First distribution is significantly ' + alt + ' than the second - ' + str(p_2))
                else: print('First distribution is NOT significantly ' + alt + ' than the second')

    # More than two distributions to compare
    else:
        # Two-sided statistical test
        if flag == 'two-sided':
            # Parametric statistical test
            if count_no_norm == 0: s_3, p_3 = f_oneway(m[0], m[1], m[2])
            # Non-parametric statistical test
            else: s_3, p_3 = kruskal(m[0], m[1], m[2])
            # Printing results
            if p_3 < alpha: print('There is statistical significant difference - ' + str(p_3))
            else: print('There is no statistical significant difference')

        # One-sided statistical test
        else:
            for alt in ['less', 'greater']:
                for d_1 in range(len(m)):
                    for d_2 in range(d_1 + 1, len(m)):
                        # Parametric statistical tests
                        if count_no_norm == 0: s_4, p_4 = ttest_ind(m[d_1], m[d_2], alternative=alt)
                        # Non-parametric statistical tests
                        else: s_4, p_4 = mannwhitneyu(m[d_1], m[d_2], alternative=alt)
                        # Printing results
                        if p_4 < alpha:
                            print('Distribution ' + str(d_1 + 1) + ' is significantly ' + alt +
                                  ' than distribution ' + str(d_2 + 1) + ' - ' + str(p_4))
                        else:
                            print('Distribution ' + str(d_1 + 1) + ' is NOT significantly ' + alt +
                                  ' than distribution ' + str(d_2 + 1))


def jsd_fsi_statistic(jsd_fsi_path_ear, alpha=0.05):
    # Function to compare the JSD-FSI scores related to PSG-to-In-ear-EEG comparisons for the different sleep stages
    # Input:
    # - jsd_fsi_path_ear: absolute path to the folder containing JSD-FSI scores for the PSG-to-In-ear-EEG comparison
    # - alpha: significance level to use for statistical tests

    classes = [os.path.basename(file).split('.')[0].split('_')[-1]
               for file in glob.glob(jsd_fsi_path_ear + '/JSD_FSI_*.npy')]

    m2_ind = np.load(os.path.join(jsd_fsi_path_ear, 'M2_index.npy')).squeeze().tolist()
    # In a matrix with dimensions [Number of comparisons X Number of subjects], the cells to exclude are [12, 2] and
    # [12, 5] i.e., the JSD-FSI for the channel M2 related to subjects 3 and 6
    # Once flatten the matrix, the indexes to exclude must be evaluated as [(ind_row x N_col) + ind_col]

    jsd_fsi = []
    for nc, c in enumerate(classes):
        print('Distribution ' + str(nc) + ' - ' + c)
        matrix = np.load(glob.glob(jsd_fsi_path_ear + '/JSD_FSI_' + c + '.npy')[0]).squeeze()

        if nc == 2:
            matrix = np.delete(matrix, [2, 7], axis=1)
            m2_flat = (m2_ind * np.size(matrix, 1)) + 5
        else:
            m2_flat = [(m2_ind * np.size(matrix, 1)) + 2, (m2_ind * np.size(matrix, 1)) + 5]
        jsd_fsi.append(np.delete(matrix.flatten(), m2_flat))

        avg_jsd_fsi = np.round(np.mean(np.delete(matrix.flatten(), m2_flat)), 2)
        std_jsd_fsi = np.round(np.std(np.delete(matrix.flatten(), m2_flat)), 2)
        print("Average JSD-FSI for {}: {} \u00B1 {}".format(c, avg_jsd_fsi, std_jsd_fsi))

    statistical_analysis(jsd_fsi)
    statistical_analysis(jsd_fsi, flag='one-sided')
