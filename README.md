# VAEGAN

The paper describing the Vaegan approach will appear at GLSVLSI 2024.

### Overview
This repository contains the code and supporting documents for the VAEGAN.

### Features
- **Metrics**: Includes the evaluation metrics such as Maximum Mean Discrepancy (MMD), Sum of Squared Differences (SSD), Precision-Recall Density (PRD), and Cosine Similarity (COSS) for both MLPVAE and DCGAN models.

- **CustomDataset**: Manages the creation of batches of files for the networks, facilitating efficient data handling.

- **DataTransformation**: Transforms float and integer numbers into a 32-bit fixed-point binary number, with 12 bits allocated for the fractional part, optimizing data representation for computational processes.

### Usage
Details on how to use these modules and scripts will be provided in subsequent sections or documents within this repository.

### Contact
For more information or queries, please contact Yuchao Liao at [yuchaoliao@arizona.edu](mailto:yuchaoliao@arizona.edu).

### Citing Vaegan
If you use the Vaegan in your work, please cite it as follows:

```
@article{liao2024skip,
  title={Skip the Benchmark: Generating System-Level High-Level Synthesis Data using Generative Machine Learning},
  author={Liao, Yuchao and Adegbija, Tosiron and Lysecky, Roman and Tandon, Ravi},
  journal={arXiv preprint arXiv:2404.14754},
  year={2024}
}
```

