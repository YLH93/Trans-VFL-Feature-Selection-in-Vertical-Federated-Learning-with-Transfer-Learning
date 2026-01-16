# Trans-VFL: Communication-Efficient Learned Feature Selection in Vertical Federated Learning

**Trans-VFL** is a novel privacy-preserving framework designed to perform effective feature selection on high-dimensional, unstructured data (e.g., images, clinical notes) within Vertical Federated Learning (VFL) systems. By integrating deep transfer learning with a communication-efficient pruning mechanism, Trans-VFL addresses the "curse of dimensionality" while minimizing data leakage.

## 📄 Abstract

[cite_start]Vertical Federated Learning (VFL) allows organizations to collaboratively train models on partitioned data without sharing raw information[cite: 4, 10]. [cite_start]However, standard feature selection methods struggle with unstructured data where individual raw features (e.g., pixels) lack semantic meaning[cite: 31].

[cite_start]**Trans-VFL** bridges this gap by shifting the selection focus from the raw input layer to the semantically rich embedding layer of a frozen, pre-trained base model[cite: 33]. [cite_start]It employs a three-stage training protocol culminating in **Local Learned Feature Selection**, where each site independently prunes non-significant features using Group Lasso regularization[cite: 37, 42]. [cite_start]This approach incurs **zero communication cost** during the selection phase and enhances privacy by keeping feature importance profiles local[cite: 164, 175].

## 🚀 Key Features

* [cite_start]**Deep Transfer Learning Integration:** Leverages frozen pre-trained models (e.g., ResNet, BERT) to extract high-level semantic features from unstructured data[cite: 34, 35].
* [cite_start]**Communication-Efficient:** Shifts the computational burden from the network to local sites[cite: 231]. [cite_start]The core feature selection stage requires **no data exchange** with the server[cite: 137].
* [cite_start]**Privacy-Preserving:** Raw data is never transmitted[cite: 171]. [cite_start]Feature pruning decisions are made locally, concealing which specific biomarkers or attributes are missing or irrelevant[cite: 175, 176].
* [cite_start]**High Sparsity & Interpretability:** Capable of reducing complex feature spaces to a small set of highly predictive signals (e.g., reducing 120 clinical variables to 28)[cite: 283].

## 🏗️ Architecture

[cite_start]The Trans-VFL framework consists of a central server and $M$ local parties[cite: 49]. [cite_start]The local model at each party $m$ is composed of three distinct components[cite: 52]:

1.  **Frozen Pre-trained Base ($f_{base}$):** Extracts a generic feature vector $z_m$ from raw input $x_m$. [cite_start]Parameters are frozen to ensure data abstraction[cite: 53, 98].
2.  **Trainable Selection Layer:** A dense layer that projects learned features into a task-specific space. [cite_start]Group Lasso regularization is applied here to drive columns of the weight matrix to zero[cite: 54, 108].
3.  [cite_start]**Trainable Embedding Head:** Compresses the selected features into the final embedding $e_m$ sent to the server[cite: 55, 112].

### The Three-Stage Algorithm

1.  [cite_start]**Collaborative Pre-training:** Updates only the selection layer and embedding head while keeping the base frozen[cite: 121, 122].
2.  [cite_start]**Embedding Component Selection:** Server identifies significant embedding components based on aggregated signals[cite: 125, 127].
3.  [cite_start]**Local Learned Feature Selection (Core Innovation):** Sites locally minimize a distillation loss combined with Group Lasso regularization to prune non-significant features without communicating with the server[cite: 129, 131].

## 📊 Results

Trans-VFL has been validated on both synthetic benchmarks and real-world healthcare datasets.

| Dataset | Metric | Trans-VFL Performance | Baseline (SecureBoost) | Features Selected (Trans-VFL) |
| :--- | :--- | :--- | :--- | :--- |
| [cite_start]**MIMIC-IV** (Mortality) [cite: 268, 276] | AUC | **0.874** | 0.862 | **28 / 120** |
| [cite_start]**Synthetic EHR** (Stroke) [cite: 203, 207] | Accuracy | **91.5%** | 87.4% | **18 / 50** (True: 20) |
| [cite_start]**Genomic HDLSS** ($N$=200) [cite: 235, 244] | Accuracy | **88.4%** | N/A | **500 / 5000** (Perfect Match) |

*Results derived from Table 1, Table 2, and Table 3 of the paper.*
