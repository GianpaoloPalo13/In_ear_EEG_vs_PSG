
from argparse import ArgumentParser
from utilis import *
warnings.filterwarnings('ignore')


def get_args():
    parser = ArgumentParser(description='Evaluation of similarity between In-ear-EEG and PSG derivations')
    parser.add_argument("--PSG_data_path", type=str,
                        help='Absolute path to the folder containing the PSG signals in .mat format')
    parser.add_argument("--GDK_data_path", type=str,
                        help='Absolute path to the folder containing the in-ear-EEG signals in .mat format')
    parser.add_argument("--PSG_scorer_path", type=str,
                        help='Absolute path to the folder containing the PSG scorers in .mat format. The folder must '
                             'contain one sub-folder for each subject, and each of these must show a sub-folder for'
                             'each scorer')
    parser.add_argument("--GDK_scorer_path", type=str,
                        help='Absolute path to the folder containing the in-ear-EEG scorers in .mat format. The folder'
                             ' must contain one sub-folder for each subject, and each of these must show a sub-folder '
                             'for each scorer')
    parser.add_argument("--save_path", type=str,
                        help='Absolute path where to save the results')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Defining the folder where to save PSG data
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    ################################################################################################################
    # -----------------------------------------  Hypnogram-based approach  -----------------------------------------
    ################################################################################################################
    print('\nStarting the hypnogram-based approach')

    print('\nUniforming PSG and in-ear-EEG scorers')
    new_path_psg, new_path_gdk, epc_path = uniform_scorers(args.PSG_scorer_path, args.GDK_scorer_path, args.save_path)

    print('\nInter-scorer variability analysis')
    # Inter-scorer variability analysis
    psg_fleiss = inter_variability(new_path_psg)
    gdk_fleiss = inter_variability(new_path_gdk)
    # Statistical analysis
    statistical_analysis([psg_fleiss, gdk_fleiss], flag='one-sided')
    plot_inter(psg_fleiss, gdk_fleiss, args.save_path)

    print('\nIntra-scorer variability analysis')
    # Intra-scorer variability analysis
    psg_gdk_cohen = intra_variability(new_path_psg, new_path_gdk)
    # Statistical analysis
    statistical_analysis([psg_gdk_cohen[i, :] for i in range(np.size(psg_gdk_cohen, 0))])
    plot_intra(psg_gdk_cohen, args.save_path)

    ################################################################################################################
    # ------------------------------------------  Feature-based approach  ------------------------------------------
    ################################################################################################################
    print('\nStarting the feature-based approach')

    # ----------------------------------------  Ground truth label-reference  ---------------------------------------

    print('\nDefining the scoring reference')
    # Consensus definition and reference evaluation i.e., intersection between PSG and in-ear-EEG consensus
    PSG_consensus_path = consensus_definition(new_path_psg, args.save_path, True)
    GDK_consensus_path = consensus_definition(new_path_gdk, args.save_path, True)
    inter_path = intersection(PSG_consensus_path, GDK_consensus_path, args.save_path)

    # --------------------------------------------  Feature extraction  --------------------------------------------
    
    print('\nStarting feature extraction')
    feat_path = feature_extraction(args.PSG_data_path, args.GDK_data_path, epc_path, args.save_path)

    # --------------------------------------------  Handling NaN values  -------------------------------------------

    print('\nHandling NaN values')
    handle_nan(feat_path, inter_path, flag=True, save_path=args.save_path)
    # Notes: 1) By analysing the feature values from the channels of all subjects, we found different channels showing
    # NaN values on just one epoch in subjects 1 and 8, thus we remove the compromised epoch from all channels of such
    # subjects; 2) the unipolar derivation M2 for subjects 3 and 6 show lots of NaN values, thus it gets removed and it
    # is not considered for following analyses

    # --------------------------------------------  Feature selection  ---------------------------------------------

    print('\nStarting feature selection')
    # In-ear-EEG-to-PSG comparison analysis
    sel_feat_path_ear_psg = feature_selection(inter_path, feat_path, flag='ear_vs_psg')
    plot_most_selected(sel_feat_path_ear_psg, args.save_path)
    # PSG-to-PSG comparison analysis
    sel_feat_path_psg_psg = feature_selection(inter_path, feat_path, flag='psg_vs_psg')

    # -----------------------------------------------  JSD-FSI scores  -------------------------------------------------

    print('\nDefinition of JSD-FSI scores')
    # In-ear-EEG vs PSG scorers
    jsd_fsi_path_ear = jsd_fsi_scores(feat_path, sel_feat_path_ear_psg, inter_path, args.save_path, flag='ear_vs_psg')
    # Plot for the spatial distribution of the JSD-FSI scores for In-ear-EEG vs PSG
    head_plot(jsd_fsi_path_ear, args.save_path)
    jsd_fsi_statistic(jsd_fsi_path_ear)

    # PSG vs PSG scorers
    jsd_fsi_path_psg = jsd_fsi_scores(feat_path, sel_feat_path_psg_psg, inter_path, args.save_path, flag='psg_vs_psg')

    # Plot for comparing JSD-FSI scores between in-ear-EEG-to-PSG and PSG-to-PSG
    hist_plot(jsd_fsi_path_ear, jsd_fsi_path_psg, args.save_path)
    # Plot to separately analyse the JSD-FSI scores for (in-ear-EEG vs scalp-EEG), (in-ear-EEG vs EOG), (EOG vs EOG),
    # and (scalp-EEG vs scalp-EEG)
    separate_hist_plot(jsd_fsi_path_ear, jsd_fsi_path_psg, args.save_path)
