# Comparison analysis between standard polysomnographic data and in-ear-EEG signals: A preliminary study
This study aims to establish a methodology to assess the similarity between the single-channel in-ear EEG signal and standard PSG derivations. Please read our accompanying paper for details [https://arxiv.org/abs/2401.10107].


## Methods
The recordings are analyzed following two approaches:

1. **Hypnogram-based approach**, intended to evaluate the agreement between PSG and in-ear-EEG-derived hypnograms. This analysis relies on four steps i.e., (i) for each subject, defining PSG and in-ear-EEG consensus in a multi-source-scored dataset; (ii) measuring the PSG and in-ear-EEG consensus agreement; and assessing (iii) intra- and (iv) inter- scorer variability by comparing pairs of hypnograms scored by the same sleep expert (PSG-to-In-ear-EEG) and groups of hypnograms referring to the same signal source (PSG-to-PSG and In-ear-EEG-to-In-ear-EEG). The sleep stages analysed in the study are 'Awake', 'NREM', and 'REM'.

2. **Feature-based approach**, based on (i) time- and frequency- domain feature extraction; (ii) unsupervised feature selection; and (iii) comparison of the distributions of the selected features via Jensen-Shannon divergence to define the Feature-based Similarity Index (JSD-FSI). This analysis is carried out first between PSG and in-ear-EEG signals (PSG-to-In-ear-EEG comparison), and then among all the possible combinations of PSG derived signals (PSG-to-PSG comparison).



## Dataset
### Data folders
Our dataset includes 21 unipolar and bipolar PSG derivations along with a single-channel in-ear-EEG.
In order to execute the code successfully, ensure that the following four folders are present:
* *'GDK_data'*, which should contain the pre-processed in-ear EEG signals in .mat format
* *'GDK_scorers'*, which should contain the associated hypnograms in .mat format for the in-ear EEG signals
* *'PSG_data'*, which should contain the pre-processed PSG signals in .mat format
* *'PSG_scorers'*, which should contain the associated hypnograms in .mat format for the PSG signals

The structure for the folders containing in-ear-EEG/PSG signals i.e., *'GDK_data'* and *'PSG_data'* must be as follow:  
<pre>
├── GDK_data                   ├── PSG_data
│ ├── Subject_01_GDK.mat       │ ├── Subject_01_PSG.mat
│ ├── Subject_02_GDK.mat       │ ├── Subject_02_PSG.mat
│ ├──   . . .                  │ ├──   . . .
│ └── Subject_10_GDK.mat       │ └── Subject_10_PSG.mat
</pre>

The structure for the folders containing the scorers for in-ear-EEG/PSG signals i.e., *'GDK_scorers'* and *'PSG_scorers'* must be as follow:  
<pre>
├── GDK_scorers                     ├── PSG_scorers  
│ │                                 │ │
│ ├── Subject_01                    │ ├── Subject_01
│ │ ├── scorer_1:                   │ │ ├── scorer_1:
│ │ │ └── Subject_01.mat            │ │ │ └── Subject_01.mat
│ │ ├── scorer_2:                   │ │ ├── scorer_2:
│ │ │ └── Subject_01.mat            │ │ │ └── Subject_01.mat
│ │ ├── scorer_3:                   │ │ ├── scorer_3:
│ │ │ └── Subject_01.mat            │ │ │ └── Subject_01.mat
│ │                                 │ │
│ ├── Subject_02                    │ ├── Subject_02
│ │ ├──  . . .                      │ │ ├──  . . .
│ │                                 │ │
│ ├──  . . .                        │ ├──  . . .
│ │                                 │ │
│ ├── Subject_10                    │ ├── Subject_10
│ │ ├──  . . .                      │ │ ├──  . . .
</pre>


### Data structure
- For both <ins>in-ear-EEG and PSG hypnogram</ins>: each *.mat* file must show one field 'stages' containing all the sleep stages (1-D array).
  
- For <ins>in-ear-EEG data</ins>: each *.mat* file must show two fields:
  1. *'GDK'*, containing the in-ear-EEG signal (1-D array)
  2. *'GDK_name'*, containing the name of the in-ear-EEG channel (list e.g. ['ch1']).  

- For <ins>PSG data</ins>: each *.mat* file must show eight fields:
  1. *'EEG_bi'*, containing the signals for N1-bipolar EEG derivations (matrix N1 x M)
  2. *'EEG_bi_names'*, containing the names of the N1-bipolar EEG derivations (list e.g. ['F3M2', 'C3M2', 'O1M2', 'F4M1', 'C4M1', 'O2M1'])
  3. 'EOG_bi', containing the signals for N2-bipolar EOG derivations (matrix N2 x M)
  4. 'EOG_bi_names', containing the names of the N2-bipolar EOG derivations (list e.g. ['E1M2', 'E2M2', 'E2M1', 'E1M1'])
  5. 'EEG_uni', containing the signals for N3-unipolar EEG derivations (matrix N3 x M)
  6. 'EEG_uni_names', containing the names of the N3-unipolar EEG derivations (list e.g. ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'M1', 'M2'])
  7. 'EOG_uni', containing the signals for N4-unipolar EOG derivations (matrix N4 x M)
  8. 'EOG_uni_names', containing the names of the N4-unipolar EOG derivations (list e.g. ['E1', 'E2']). M is the number of samples.  

  > [!IMPORTANT]
  > An additional <ins>*mastoid-to-mastoid derivation*</ins> ('M2M1') gets automatically defined in the script.



## Usage
The main script to execute is *main.py*: all the functions that are used are defined in *utils.py*. To run the script, please use the following syntax:
```blue
python main.py --PSG_data_path r'.\In_ear_EEG_vs_PSG\PSG_data' --GDK_data_path r'.\In_ear_EEG_vs_PSG\GDK_data' --PSG_scorer_path r'.\In_ear_EEG_vs_PSG\PSG_scorers' --GDK_scorer_path r'.\In_ear_EEG_vs_PSG\GDK_scorers' --save_path r'.\In_ear_EEG_vs_PSG\Results'
```
