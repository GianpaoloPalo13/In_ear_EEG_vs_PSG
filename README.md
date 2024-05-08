# In_ear_EEG_vs_PSG
Github repository related to the paper "Comparison analysis between standard polysomnographic data and in-ear-EEG signals: A preliminary study".
In this study, the dataset includes 21 unipolar and bipolar PSG derivations along with a single-channel in-ear-EEG; and the sleep stages analysed are 'Awake', 'NREM', and 'REM'.

In order to execute the code successfully, ensure that the following four folders are present:
* 'GDK_data', which should contain the in-ear EEG signals in .mat format
* 'GDK_scorers', which should contain the associated hypnograms in .mat format for the in-ear EEG signals
* 'PSG_data', which should contain the PSG signals in .mat format
* 'PSG_scorers', which should contain the associated hypnograms in .mat format for the PSG signals

Each .mat file for the in-ear-EEG data must show two fields i.e., 1) 'GDK', containing the in-ear-EEG signal (1-D array); and 2) 'GDK_name', containing the name of the in-ear-EEG channel (list e.g. ['ch1']).
Each .mat file for the PSG data must show eight fields i.e., 1) 'EEG_bi', containing the signals for N1-bipolar EEG derivations (matrix N1 x M); 2) 'EEG_bi_names', containing the names of the N1-bipolar EEG derivations (list e.g. ['F3M2', 'C3M2', 'O1M2', 'F4M1', 'C4M1', 'O2M1']); 3) 'EOG_bi', containing the signals for N2-bipolar EOG derivations (matrix N2 x M); 4) 'EOG_bi_names', containing the names of the N2-bipolar EOG derivations (list e.g. ['E1M2', 'E2M2', 'E2M1', 'E1M1']); 5) 'EEG_uni', containing the signals for N3-unipolar EEG derivations (matrix N3 x M); 6) 'EEG_uni_names', containing the names of the N3-unipolar EEG derivations (list e.g. ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'M1', 'M2']); 7) 'EOG_uni', containing the signals for N4-unipolar EOG derivations (matrix N4 x M); 8) 'EOG_uni_names', containing the names of the N4-unipolar EOG derivations (list e.g. ['E1', 'E2']). M is the number of samples.
Note: the mastoid-to-mastoid derivation ('M2M1') described in the paper is automatically defined in the script.

The structure for the folders containing the in-ear-EEG/PSG signals must be as follow:
- GDK_data
  - Subject_01_GDK.mat
  - Subject_02_GDK.mat
  - ...
  - Subject_10_GDK.mat

Each .mat file for the in-ear-EEG/PSG hypnogram must show a field 'stages' containing all the sleep stages (1-D array).

The structure for the folders containing the scorers for in-ear-EEG/PSG signals must be as follow:
- GDK_scorers
  - Subject_01
    - scorer_1:
      - Subject_01.mat
    - scorer_2:
      - Subject_01.mat
    - scorer_3:
      - Subject_01.mat 
  - Subject_02
    - ... 
  - ...
  - Subject_10
    - ...
