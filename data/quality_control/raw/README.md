# PharmaVision pharmaceutical tablet image dataset

- **Source:** [Zenodo record 22773411](https://zenodo.org/records/22773411)
- **Official research/code:** [pharmaceutical-tablet-defect-detection](https://github.com/tahirlee/pharmaceutical-tablet-defect-detection)
- **Archive:** PharmaVision.rar
- **Dataset:** 640 original images; normal and defective classes
- **Original split counts:** Train 190/class, Val 30/class, test 100/class
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0), as recorded in Zenodo metadata
- **Paper:** Mahmood et al., “Artificial Intelligence-Based Quality Control of Pharmaceutical Tablets: A Visual Dataset and Benchmark for Automated Defect Detection,” Pharmaceutics 18(10), 1287 (2026), DOI: <https://doi.org/10.3390/pharmaceutics18101287>

The training pipeline downloads the Zenodo archive into this directory and
extracts the original fixed Train, Val, and test folders beneath
`PharmaVision/`. It verifies Zenodo's published archive checksum and checks
the expected class directories and image counts.

This dataset supports visual normal-versus-defective classification only.
Model output and Grad-CAM do not verify chemical composition, medicine
identity, dosage, pharmaceutical quality, or patient safety.
