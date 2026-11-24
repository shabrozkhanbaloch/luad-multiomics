# Table 2. Somatic mutation block (TCGA-LUAD, MC3)

512 patients · 16949 genes mutated in ≥1 patient · 136,256 non-silent variants

## Genes altered in ≥5% of the cohort (top 25 of 702)

> Reported for description only. Both the threshold and the top-K cap are refitted inside each training fold, so the modelled set varies by fold.

> **702 genes clear 5%, not the ~26 the protocol assumed.** LUAD's mutation burden (median 4.8 mut/Mb) puts that many binary features at EPV 0.30 against 210 PFI events, and most are length-biased passengers. The block is capped at the 25 most frequent; driver biology enters through the pathway features below.

| Gene     |   n altered | % of cohort   |
|:---------|------------:|:--------------|
| TP53     |         259 | 50.6%         |
| TTN      |         244 | 47.7%         |
| MUC16    |         213 | 41.6%         |
| CSMD3    |         204 | 39.8%         |
| RYR2     |         190 | 37.1%         |
| LRP1B    |         172 | 33.6%         |
| USH2A    |         160 | 31.2%         |
| ZFHX4    |         157 | 30.7%         |
| KRAS     |         151 | 29.5%         |
| FLG      |         130 | 25.4%         |
| SPTA1    |         129 | 25.2%         |
| XIRP2    |         129 | 25.2%         |
| ZNF536   |         105 | 20.5%         |
| FAT3     |         104 | 20.3%         |
| COL11A1  |         103 | 20.1%         |
| CSMD1    |         102 | 19.9%         |
| NAV3     |         102 | 19.9%         |
| MUC17    |         102 | 19.9%         |
| PCDH15   |         100 | 19.5%         |
| ANK2     |          98 | 19.1%         |
| PAPPA2   |          95 | 18.6%         |
| PCLO     |          94 | 18.4%         |
| KEAP1    |          93 | 18.2%         |
| ADAMTS12 |          93 | 18.2%         |
| APOB     |          90 | 17.6%         |

## Oncogenic pathway alteration (Sanchez-Vega et al. 2018)

| Pathway    |   Genes in list |   Genes mutated in cohort |   Patients altered | % altered   |
|:-----------|----------------:|--------------------------:|-------------------:|:------------|
| cell_cycle |              15 |                        12 |                 69 | 13.5%       |
| hippo      |              38 |                        36 |                317 | 61.9%       |
| myc        |              13 |                        11 |                 72 | 14.1%       |
| notch      |              71 |                        62 |                269 | 52.5%       |
| nrf2       |               3 |                         3 |                114 | 22.3%       |
| pi3k       |              29 |                        28 |                217 | 42.4%       |
| rtk_ras    |              85 |                        82 |                446 | 87.1%       |
| tgf_beta   |               7 |                         7 |                 57 | 11.1%       |
| tp53       |               6 |                         6 |                294 | 57.4%       |
| wnt        |              68 |                        64 |                259 | 50.6%       |
