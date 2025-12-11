# Deep Learning GNN (MTI–GAT): Structure & Function-Aware Multi-Omic Graph Attention Network for Gene Prioritisation


This repository implements a **deep learning framework** built on a supervised **Graph Attention Network (GAT)** to prioritise exercise-responsive and ageing-related genes.  

The model integrates:

- **Mendelian randomisation (MR) causal signals** across four omics,
- the **multi-omic MTI (Modfied Trait Importance) score**,
- **AlphaFold v6 structural descriptors**,
- **PANTHER functional annotations** (4 layers),
- **graph topology metrics** extracted from a kNN gene–gene network.

The GAT learns structure and function-aware MTI embeddings and refines gene ranking beyond raw MR/MTI values.

---

## Core scoring basis

Gene prioritisation is driven by:

### **1. MR-derived causal evidence**
MR betas, SE, p-values, FDR, causal scores, nsnp, and layer-specific MR statistics form the core feature set.

### **2. The multi-omic MTI score**
Serves as the **supervised regression target**, representing integrated causal support across proteomic, CpG, glycan and single-cell transcriptomic data.

Thus, **MR + MTI are the fundamental quantitative basis for all model predictions**, while structural, functional, and topological features modulate the deep-learning representation.

---

## Feature groups

### **AlphaFold2 structure features**
- pLDDT distributions  
- secondary structure fractions  
- disorder/exposure metrics  
- domain and interface descriptors  

### **PANTHER functional structure (4 layers)**
- protein class  
- molecular function  
- biological process  
- pathway / component annotations  

### **Graph topology**
Derived from the kNN gene-similarity graph (k=15):
- degree  
- PageRank  
- eigenvector centrality  
- betweenness  
- closeness  
- clustering coefficient  

---

## Model architecture

- **Two-layer GAT encoder** (`GATConv`, ELU, dropout)
- **Regression head:** MTI prediction  
- **Classification head:** multi-layer support (`MTI_n_layers ≥ 2`)
- **Loss:**  
  `L = MSE(MTI_score) + λ × BCE(multi-layer label)`

---

## Inputs

### Node table (`--node_path`)
Must contain:
- `gene_symbol`
- `MTI_score`

Used as features:
- all MR numeric features  
- single-cell MR features  
- AlphaFold2 structure metrics  
- PANTHER annotations  
- topology and structural-summary features  

### Edge table (`--edge_path`)
kNN edges with `from` / `to` or autodetected equivalents.

---

## Outputs

Writes a TSV with:
- MTI-GNN predictions  
- ranks + scaled ranks  
- multi-layer probabilities  
- learned embeddings  

---

## Summary

**Deep learning meets multi-omic causal inference.**  
This model uses MR + MTI as its primary evidence base and incorporates AlphaFold2, PANTHER and graph topology to achieve **structure-aware, biologically contextualised gene prioritisation**.
