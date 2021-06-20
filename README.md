# Travel behaviour change detection study

## Code Structure
The main entrance for SBB and Geolife datasets:
- main_SBB: the whole pipeline for the SBB dataset.
- main_Geolife: the whole pipeline for the Geolife dataset.

Files containing the different steps of the pipeline:
- getActivitySet.py: generate activity set and important trip set
- similarityMeasures.py: similarity measurement 
- clustering.py: clustering 
- clusterVisualization.py: clustering result analysis and plot
- changeDetection.py: change detection algorithms and result plot
- jupyter notebook scripts:
    - stat.ipynb: get preprocessed data size, prove of stability for important trip set, and top1 location change detection (proxy for home change)
    - tracking_quality.ipynb: select users based on tracking coverage.
- And helper script in .utils/ folder:
    - config.py: define data paths for intermediate results.
    - data_figure.py: helper function to generate data for Figure 2.
    - generateLocation.py: location generation from stay points.
    - preProSBB.py: data loading and preprocessing for the SBB dataset.
    - preProGeolife.py: data loading and preprocessing for the Geolife dataset.

## User selection for SBB
Users are pre-filtered based on overall and sliding window tracking quality
- user tracked > 300 days.
- for each time window of 10 weeks, user tracking quality > 0.6.

All time-series cut at 2017-12-25 when the main study ends. 
- for demonstrating cluster result (Figure 3): user 1659.
- for demonstrating change detection results (Figure 4): user 1659.
- for comparing different users (Figure 5): (A) user 1632, (B) user 1641, (C) user 1620, and (D) user 1630.

Users who changed their top1 location during the study (a proxy for home location change):
- 1 time: user 1651, 1624, 1608
- 2 times: user 1650 (probably holiday house), 1620 (intercontinental travel, probably business reasons)
- Multiple times (probably multiple homes/holiday house): user 1631, 1630

## Requirements and dependencies
* Numpy
* GeoPandas
* Matplotlib 
* trackintel
* tqdm
* OSMnx