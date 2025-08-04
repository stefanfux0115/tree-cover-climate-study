# Tree Cover Energy Analysis - China

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![R](https://img.shields.io/badge/R-4.0%2B-blue)](https://r-project.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive analysis pipeline for quantifying how tree cover affects electricity consumption under different temperature conditions across China (2012-2019).

## Project Overview

This project analyzes **11.65 million grid-month observations** to understand how urban tree cover can reduce energy consumption during heat waves. Using advanced econometric methods, we quantify the cooling benefits of trees and identify where green infrastructure investments would be most effective.

### What This Project Does

- **Processes massive climate and energy datasets** (>100GB) at 1km resolution
- **Quantifies tree cooling benefits** with precise economic metrics
- **Identifies optimal locations** for urban forestry investments
- **Tests climate conditions** where trees are most/least effective
- **Provides reproducible analysis pipeline** for other regions/countries

### Key Results

- **33-48% reduction** in temperature-induced electricity consumption in tree-covered areas
- **Peak effectiveness** at 30-35°C (48% cooling benefit)
- **Reduced effectiveness** under high humidity conditions (transpiration failure)
- **Quantified energy savings** for cost-benefit analysis of green infrastructure

## Technical Stack

### Languages & Tools
- **Python 3.8+**: Data processing, ML experiments
- **R 4.0+**: Econometric analysis, visualization
- **Git**: Version control

### Key Libraries
- **Python**: pandas, numpy, scikit-learn, lightgbm, xgboost, shap
- **R**: fixest, ggplot2, sf, raster, dplyr

### Data Processing
- **Spatial Resolution**: 1km × 1km grid cells
- **Temporal Coverage**: Monthly data from 2012-2019
- **Data Volume**: >100GB raw data processed

## Project Structure

```
code-test/
├── scripts/                     # Analysis pipeline
│   ├── Download-Cleaning/       # Data acquisition scripts
│   ├── Data-Processing/         # Data integration & processing  
│   └── ML-Analysis/             # Machine learning experiments
├── config/                      # Configuration files
│   ├── download-config.json     # API keys and download settings
│   ├── data-processing.json     # Processing parameters
│   └── ml-analysis.json         # ML model configurations
├── data/                        # Data storage (contents gitignored)
│   ├── raw/                     # Original downloaded datasets
│   ├── processed/               # Processed analysis-ready data
│   └── shp/                     # Shapefiles for geographic boundaries
├── r-analysis/                  # R econometric analysis
│   ├── geo-analysis.Rmd         # Main analysis notebook
│   ├── results/                 # Regression output files
│   └── figs/                    # Generated figures
├── output/                      # Analysis outputs (gitignored)
│   ├── feature_selection/       # Feature selection results
│   ├── ml_pipeline/             # ML model results
│   └── shap_analysis/           # SHAP interpretability results
├── logs/                        # Processing logs (gitignored)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

**Note**: The `data/`, `output/`, and `logs/` directories contain large files that are not included in the repository. You'll need to generate these by running the data download and processing scripts.

## Quick Start

### 1. Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt

# R 4.0+ with required packages
install.packages(c("fixest", "ggplot2", "sf", "raster", "dplyr"))
```

### 2. Clone Repository

```bash
git clone https://github.com/stefanfux0115/code-test.git
cd code-test
```

### 3. Configure Data Paths

Edit the JSON files in `config/` directory to set your local data paths and API keys.

### 4. Run Analysis Pipeline

```bash
# Download raw data (requires API keys)
python scripts/Download-Cleaning/download_master.py --all

# Process and integrate data
python scripts/Data-Processing/unified_data_processor_optimized.py --all

# Run econometric analysis
# Open r-analysis/geo-analysis.Rmd in RStudio
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Author

**Enxian Fu**  
Email: [stefanfu015@gmail.com](mailto:stefanfu015@gmail.com)

## Acknowledgments

- Research advisors and collaborators
- Data providers (ERA5, Chen et al., ChinaMet)
- Open source community

---

**Star this repo if you find it useful!**

**Found a bug?** Open an issue!

**Have questions?** Feel free to reach out!
