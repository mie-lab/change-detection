# Travel behaviour change detection study

# Structure
- 00_preProSBB.py: load data and preprocessing
- 02_activitySet.py: generate activity set and important trip set
- 03_similarity.py: similarity measurement 
- 04_cluster.py: clustering 
- 05_analysis.py: clustering result analysis and plot
- 06_change.py: change detection algorithms and result plot
- And jupyter notebook helper script:
    - basic_stat.ipynb: get preprocessed data size, get intermodal trip number, and top1 location change detection (proxy for home change)
    - intermodal_trip.ipynb: extract intermodal trips and visualize
    - select_quality_stability.ipynb: select users based on tracking coverage, and prove of stability for (activity set and) important trip set.
    - figure\data_figure.py: preprocessing for generating the tripleg data figure.

# User selection
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