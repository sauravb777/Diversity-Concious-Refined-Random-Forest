# Diversity Conscious Refined Random Forest (DCRRF) — Implementation

This repository contains an implementation of **Diversity Conscious Refined Random Forest (DCRRF)** — a Refined Random Forest classifier that iteratively prunes unimportant features, grows trees only on informative features, and retains a maximally diverse final ensemble by clustering and removing correlated trees. The approach and algorithm are described in the original paper: *Diversity Conscious Refined Random Forest* (arXiv). :contentReference[oaicite:0]{index=0}

## Summary
DCRRF aims to reduce redundancy and inference cost in Random Forests by:
- Iteratively removing least-informative features (feature refinement).
- Analytically deciding how many new trees to add per iteration.
- Measuring per-tree AUCs and clustering trees by pairwise correlation, then selecting the top-performing (highest AUC) uncorrelated tree from each cluster to form a compact, diverse ensemble. :contentReference[oaicite:2]{index=2}
