# Prostate Radiotherapy Segmentation: 2D vs 3D - Assessing Benefits of Volumetric Context

Author: Bhaveshyien Thyaganathan - 48426835

## Algorithm and Problem statement
### The problem:
During treatment, the prostate must be specifically targetted, if the beam is over/under contoured there is risk of irradiating patient tissue. Therefore, before treatment, the prostate must be accurately segmented from healthy tissue such that only cancerous tissue is targeted by the radiation beam.

Furthermore, the segmentation capability of 2D models and 3D models will be evaluated in troublesome areas such as the apex and base of the tumour, wherein the tumour may account for less than a percent of total scan area.

### The solution:
This project aims to solve the above problem through the implementation of a 3D UNet architecture to solve a lack of spacial context at the base and apex of the tumour. The ability for the model to discern the tumour will be evaluated
between the 2D and 3D UNets to assess if the additional performance accuracy warrants an increase in compute and resources.

### Algorithm:
The architectures used for both the 2D and 3D models are UNets. These models involve convolving a large context down to a bottleneck where key features are identified, these features are then upscaled and mapped to features from the original through provided skip connections. The 3D network adapts these methods through 3D convolution and 3D max pooling to capture spacial continuity across multiple slices.

![3D UNet Architecture](3Darchitecture.png)

## Feasibility Review

### User need, scope and acceptance criteria:

* **Intended User**: An imaging specialist who reviews scans of prostate contours to identify malignant tumours.
* **Purpose**: Test whether 3D volumetric contexts fixes detection at the base and apex of the tumour as opposed to a 2D U-net baseline model.
* **Acceptance criteria**:

| **Criteria**                   | **Descriptor**                                                                            |
|:-------------------------------|:------------------------------------------------------------------------------------------|
| Overall accuracy               | 3D U-net test DICE coefficient ≥ 0.7                                                      |
| Dilemma criterion (boundary)   | Reports DICE separately for the base, mid and apex of each volume.                        |
| Dilemma criterion (continuity) | Quantify consistency within a slice, test for DICE drops between slices                   |
| Resource constraints           | Peak training VRAM ≤ 16Gb, and inference ≤ 10s (per vol.) on a single rangpur GPU (A100). |
| Data Hygiene                   | Zero patient overlap between train/validation/test splits.                                |

### Model choice and course concepts:

* **Data**: Down sampled HipMRI 3D loaded with the Nifti loader in Appendix B. Labels are one-hot encoded, and images are z-score normalised.
* **Model**: 3D U-Net *[Çiçek et al., 2016]*: a 3D encoder–decoder with skip connections, plus instance/batch norm, ReLU, and a Dice + cross-entropy loss to handle the heavy class imbalance.
* **Baseline**: A 2D U-Net *[Ronneberger et al., 2015]* trained on the 2D slices from the same patients, with predictions stacked into volumes and evaluated on the identical test split.
* **Course concepts**: Convolutional feature hierarchies, skip connections, overlap-based losses for imbalanced segmentation, data augmentation, and the memory/compute trade-off of dimensionality.
* **Split**: Patient-level split (e.g. [28 / 5 / 5] patients), because one patient contributes multiple volumes and slice-level splitting would leak anatomy.

### Risks, budget and fallback:

| **Risk**                                     | **Mitigation**                                                               |
|:---------------------------------------------|:-----------------------------------------------------------------------------|
| GPU memory limits at full volume size        | Smaller base filters, batch size 1, mixed precision, or patch-based training |
| Class imbalance (leads to empty predictions) | Dice/Focal loss, patch sampling biased toward the prostate                   |
| Rangpur queue delays                         | Short Slurm jobs with checkpointing; start early                             |

* **Next experiment**: Full 3D vs 2D comparison on the held-out split, with per-region (base/mid/apex) Dice.
* **Fallback**: If full-volume 3D training does not fit in memory or converge, switch to patch-based 3D training. If the 3D model still misses Dice 0.70, report the result honestly with a failure analysis. The dilemma investigation (where context helps and where it doesn’t) still earns marks.
