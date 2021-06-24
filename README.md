# Travel behaviour change detection
This repository represents the implementation of the paper:

### [A Clustering-Based Framework For Individual Travel Behaviour Change Detection]()
[Ye Hong](https://scholar.google.com/citations?user=dnaRSnwAAAAJ&hl=en), [Yanan Xin](https://baug.ethz.ch/en/department/people/staff/personen-detail.Mjc4MjA5.TGlzdC82NzksLTU1NTc1NDEwMQ==.html), [Henry Martin](https://n.ethz.ch/~martinhe/), [Dominik Bucher](https://scholar.google.ch/citations?user=15XEBsQAAAAJ&hl=de), [Martin Raubal](https://raubal.ethz.ch/)\
[IKG, ETH Zurich](https://gis.ethz.ch/en/)

![cluster_dome](figures/3_cluster/cluster_demo.png?raw=true)

## Reproduce the framework on the Geolife dataset
While the results in the paper are obtained from SBB Green Class dataset that is not publicly available, we provide a runnable example of the pipeline on the Geolife dataset. The steps to run the pipeline are as follows:
- Download the repo, install neccessary `Requirements and dependencies`.
- Download the Geolife GPS tracking dataset from [here](https://www.microsoft.com/en-us/download/details.aspx?id=52367). Unzip and copy the `Data` folder into `geolife/`. The file structure should look like `geolife/Data/000/...`.
- Define your working directories in `utils/config.py`.
- Run `utils/preProGeolife.py` and `utils/generateLocation.py` scripts to generate trips and locations.
- Run the `main_Geolife.py` script for the travel behaviour change detection pipeline. The Figures and detection results are saved in the `config["resultFig"]` folder.

**Note: this is only for demonstration purposes, and the parameter combinations are not guaranteed to produce meaningful results.**

## Code Structure
The main entrance for SBB and Geolife datasets:
- `main_SBB.py`: the whole pipeline for the SBB dataset.
- `main_Geolife.py`: the whole pipeline for the Geolife dataset.

Files containing the different steps of the pipeline:
- `getActivitySet.py`: generate activity set and important trip set
- `similarityMeasures.py`: similarity measurement 
- `clustering.py`: clustering 
- `clusterVisualization.py`: clustering result analysis and plot
- `changeDetection.py`: change detection algorithms and result plot
- jupyter notebook scripts:
    - `stat.ipynb`: get preprocessed data size, prove of stability for the important trip set, and top1 location change detection (a proxy for home changes)
    - `tracking_quality.ipynb`: select users based on tracking coverage.
- And helper script in `.utils/` folder:
    - `config.py`: define data paths for intermediate results.
    - `data_figure.py`: helper function to generate data for Figure 2.
    - `generateLocation.py`: location generation from stay points.
    - `preProSBB.py`: data loading and preprocessing (trip generation) for the SBB dataset.
    - `preProGeolife.py`: data loading and preprocessing (trip generation) for the Geolife dataset.

## User selection for SBB
Users are pre-filtered based on overall and sliding window tracking quality
- user tracked > 300 days.
- for each time window of 10 weeks, user tracking quality > 0.6.

All time-series cut at 2017-12-25 when the main study ends. 

User selection for Figures:
- for demonstrating cluster result (Figure 3): user 1659.
- for demonstrating change detection results (Figure 4): user 1659.
- for comparing different users (Figure 5): (A) user 1632, (B) user 1641, (C) user 1620, and (D) user 1630.

Users who changed their top1 location during the study (a proxy for home location change):
- for 1 time: user 1651, 1624, 1608
- for 2 times: user 1650 (probably holiday house), 1620 (intercontinental travel, probably business reasons)
- for multiple times (probably multiple homes/holiday house): user 1631, 1630

## Requirements and dependencies
* Numpy
* Pandas
* GeoPandas
* Matplotlib 
* trackintel
* tqdm
* scikit-learn-extra
