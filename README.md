# TailID

TailID is an algorithm for detecting ID-sensitive points in the tail of probability distributions for probabilistic Worst-Case Execution Time (pWCET) estimation. It has been proposed in the following paper.

```
Blau Manau, Sergi Vilardell, Isabel Serra, Enrico Mezzetti, Jaume Abella, and Francisco J. Cazorla. Detecting Low-Density Mixtures in High-Quantile Tails for pWCET Estimation. In 37th Euromicro Conference on Real-Time Systems (ECRTS 2025). Leibniz International Proceedings in Informatics (LIPIcs), Volume 335, pp. 20:1-20:25, Schloss Dagstuhl – Leibniz-Zentrum für Informatik (2025) https://doi.org/10.4230/LIPIcs.ECRTS.2025.20
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The TailID algorithm can be executed through the command line interface (CLI).

```bash
python cli.py <data_file> --p_c1 <value> --n_candidates <value> [--gamma <value>] [--mos <value>]
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `data_file` | Yes | Path to a text file containing the data to analyze (one numeric value per line) |
| `--p_c1` | Yes | Candidate percentile (0 < p_c1 < 1). Defines the starting point of the candidate point set. |
| `--n_candidates` | Yes | Number of candidate thresholds for p_m selection |
| `--gamma` | No | Confidence level (0 < gamma < 1). Controls detection sensitivity. Default: 0.9999 |
| `--mos` | No | Minimum of Samples for scenario classification. Default: 40 |

### Data File Format

The input file is a text file containing one numeric value per line.

```
1.23
4.56
7.89
...
```

#### Example

```bash
python3 cli.py example_data.txt --p_c1 0.95 --n_candidates 50
```

Example output:

```
Loaded 1000 data points from example_data.txt

Selecting optimal p_m by minimizing EQMAE...
Selected p_m = 0.8755

============================================================
TailID Analysis Result
============================================================

Parameters:
  p_m (extreme value percentile): 0.8755 (auto-selected)
  p_c1 (candidate percentile): 0.95
  n_candidates: 50
  gamma (confidence level): 0.9999
  MoS (minimum of samples): 40

Results:
  Number of sensitive points: 0
  Scenario: SCENARIO_1

Interpretation:
  Scenario 1: No inconsistent points detected (|S| = 0). The identical distribution (ID) hypothesis holds for the tail. The tail is stable and suitable for pWCET estimation. Consider combining with KPSS test for additional validation.

============================================================
```

## Discussion of Algorithm Parameters (quoted from paper)

> Algorithm 1 works on three main parameters ( $\gamma, p_{M}$, and $p_{c_{1}}$ ) that call for a careful selection to guarantee the effectiveness of the results. In the following we discuss the implications on the parameter selection and the proposed selection criteria.

> $p_{M}$. The choice of parameter $p_{M}$ determines the points in the sample that are considered part of the sample tail. Given the lack of consensus in the literature regarding the choice of percentile that defines the sample tail, several studies also criticize the practice of arbitrarily selecting a fixed percentile without first analyzing the data [43]. To address this, we determine $p_{M}$ using the methodology developed in [6] that evaluates multiple candidate percentiles and selects the one that best aligns with the assumptions of EVT. Any other threshold selection algorithm in the literature is also suitable.

> $p_{c_{1}}$. The choice of $p_{c_{1}}$ determines the points in the sample that are considered candidates for being sensitive points. This percentile is tied to the selection of $p_{M}$, such that, the candidates for sensitive points must belong to the sample tail and indeed represent data with very low probability outcomes. Additionally, as the goal of the TailID algorithm is to detect any inconsistencies in the ID of the tail, we avoid selecting a percentile $p_{c_{1}}$ that fails to detect any sensitive points while another percentile does. A Grid Search algorithm [9] employed on the kernel data, described in Section 6, has shown that choosing $p_{c_{1}}=1-\left(0.05 \cdot\left(1-p_{M}\right)\right)$ results in a stable number of detected sensitive points across different values of $p_{M}$, and across multiple scenarios. Moreover, this percentile ensures that the algorithm consistently identifies a relevant proportion of sensitive points among the candidate points, e.g. for typical thresholds between percentiles $p_{M}=[0.85,0.95]$, the candidate points selected are the top $0.75 \%$ and $0.25 \%$ of the sample respectively. Thus, we are aiming at the last component in the mixture. This choice minimizes the risk of failing to detect sensitive points when compared to other percentiles. Also, to improve robustness, we perform a grid search on the data before applying the algorithm. This preliminary step helps to determine an appropriate percentile $p_{c_{1}}$.

> $\gamma$. The confidence level $\gamma$ is set at 0.9999 to identify only those points that truly impact the tail distribution, thereby minimizing the number of false positives. For a point to be classified as sensitive, the EVI of the tail, recalculated to include this point, must fall outside the confidence interval of the EVI computed without it. With $\gamma=99.99 \%$, the risk that the true parameter of the distribution lies outside this range is $0.01 \%$.

## Decision Making on TailID outcomes (quoted from paper)

> TailID improves the interpretability of the outcomes, allowing the user to extract conclusions on the extreme values of the data at analysis. In this line, we next define several outcomes (scenarios) from the execution TailID and how to interpret them. In order to create these scenarios we consider the minimum amount of samples (MoS) required for a proper estimation of the EVI. The topic of determining the necessary amount of extreme samples to obtain an accurate estimation on the tail is outside the scope of this work. However, we refer to works in other areas where they argue that the minimum number of extreme samples varies depending on the tail profile [34, 28]. The authors in [14] develop a method to estimate the MoS for each sample where, in general, it ranges between 40 and 70 . In line with it, in this
work we conservatively set MoS to be at least $MoS=40$ and rely on [14] for further tuning. For instance, for an Extreme Value Index $\hat{\xi}=-0.5$ and $MoS=40$, the $99 \%$ confidence interval is $\hat{\xi}=-0.5 \pm 0.10$.
> Overall, using the number of inconsistent points detected, that is the cardinality of the set $S$ (i.e., $|S|$ ), we define three scenarios according to how $|S|$ compares to $MoS$ :

> Scenario 1. $(|S|=0)$ No inconsistent points detected. If the list of inconsistent points, $S$, is empty, then no candidate point fell outside the confidence intervals, and the ID hypothesis on the identical distribution of the tail holds. The TailID outcome provides more evidence on the robustness of the tail estimations given the stability of the tail. In this scenario, TailID in conjunction with the KPSS test for the stationarity on the whole distribution, provides a solid analysis on the sample

> Scenario 2. $(|S|>MoS)$ High-number of inconsistent points detected. If the final list of inconsistent points detected by the TailID algorithm is bigger than the MoS required for a proper estimation of the EVI, then we are in the presence of multiple tail behaviors, which should be taken into account when performing pWCET estimations. Hence, the sample needs to be analyzed taking into account the presence of another source of uncertainty in the tail of the distribution. In this scenario, TailID considers the threshold of the tail at the first inconsistent value found with TailID, which would be on the last mixture component. Interestingly, in the case of multiple mixture components conforming the distribution, the PoT model implies that the last component would be employed for the tail estimation, while the rest would be considered on the bulk of the distribution.

> Scenario 3. ( $|S| \leq MoS$ ) Few inconsistent points detected. As in Scenario 2 there is mixture, but in this case more samples are needed to make a better estimation of the parameters of the distribution since with less than $MoS$ points it would be imprecise. If after performing more runs $|S|>MoS$, we apply TailID as in Scenario 2. However, if the user reaches the number of runs he/she can perform - e.g. due to the associated time required to perform the runs - and still $|S| \leq MoS$, tail prediction should not be done otherwise the uncertainty in the estimation can be high.
