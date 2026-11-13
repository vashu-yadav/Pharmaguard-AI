# AI4I 2020 Predictive Maintenance Dataset

- **Source:** [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2B)
- **Download archive:** <https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip>
- **Extracted filename:** `ai4i2020.csv`
- **Format:** CSV (comma-separated text)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation:** AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. <https://doi.org/10.24432/C5HS5C>

The Smart Manufacturing pipeline downloads the official ZIP archive when this
CSV is missing, validates its schema, and saves the CSV in this directory. The
dataset is synthetic and must not be described as real factory measurements.
The pipeline retains `Machine failure` as the binary target and reports rows
where it differs from the five failure-mode flags; it does not silently rewrite
either label source. The data is used only by Smart Manufacturing and is
unrelated to the tablet image quality-control data.
