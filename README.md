# DeepQSAR-Based Antimicrobial Peptide Discovery



## 📘 Overview

This work introduces a DeepQSAR pipeline that combines a regression model and a recurrent neural network (RNN) to predict antimicrobial and anti-biofilm activity of peptides. By applying this model to 50,000 candidate sequences derived from the UniProt database, we identified 100 peptides for synthesis and validation. Among them, 29 peptides outperformed a known AMP control (IDR-1018) in both anti-biofilm and antimicrobial activities.

---

## 📁 Repository Structure

### `1_dataset/`
- Includes all training and evaluation data.
- **In-house dataset**: ~700 peptides with experimentally validated anti-biofilm IC50 values.
- **Public datasets**:
  - [DRAMP](http://dramp.cpu-bioinfor.org/)
  - [AI4AMP](https://journals.asm.org/doi/10.1128/msystems.00299-21) – see *Data Availability*
  - [DBAASP](https://dbaasp.org/home)

### `2_modeling/`
- Contains scripts and notebooks for:
  - Model 1: Regression model predicting biofilm inhibition (IC50).
  - Model 2: RNN model for antimicrobial activity prediction.
  - Combined architecture with transfer learning between Model 1 and 2.

### `3_screening/`
- Pipeline for:
  - Extracting 12-mer peptides from UniProt human proteins.
  - Filtering and scoring ~50,000 candidates using the DeepQSAR model.

### `4_clustering/`
- Clustering of top model-predicted peptides using hierarchical clustering.
- Selection of diverse peptides from each cluster for synthesis.

### `5_wetlab_results/`
- Experimental results (IC50 values) for 100 synthesized peptides.
- Includes both anti-biofilm and planktonic antimicrobial data.

### `6_plots/`
- Visualizations for:
  - Violin plots comparing training vs. novel peptide activity.
  - Scatter plots and summary tables of top peptide hits.

- **Experimental Results**:  
  - 29 peptides outperformed control in both activities.  
  - Peptide **MVLRIKLRLKIR** showed the strongest biofilm inhibition (IC50 = 0.147 µM).

