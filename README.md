# 🧠 Alzheimer's Progression via Network Diffusion Models

**Author**: Ismael Leal  
**Date**: April 2024  
**Repository**: [`alzheimers-progression`](https://github.com/IsmaLeal/alzheimers-progression)

This repository implements and visualizes network-based models of pathological tau protein spread across the human brain, simulating Alzheimer’s progression through a variety of biophysically and clinically informed regimes.

The models are based on the **FKPP (Fisher–Kolmogorov)** framework and explore the effects of clearance, volume-based scaling, and Laplacian reweighting through non-linear damage mechanisms.

---

## 📁 Project Structure

```
alzheimers-progression/
│
├── src/
│   ├── analysis/               # Experiments and analyses done in the project
│   ├── data/                   # Data loading & pre-processing
│   ├── models/                 # Mathematical model definitions & runners
│   └── visualisation/          # Graphical representations and animations
│   ├── experiments/            # Experiments reproducing paper figures
│
├── databases/                 # Connectivity & anatomical data (CSV)
├── results/                   # Output plots when using --save
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧪 Main Features

### 🧬 Models implemented
- FKPP baseline
- Coupled clearance–growth
- Linear damage (Laplacian reweighting)
- Exponential damage (Laplacian reweighting)

All models simulate tau protein spread over a structural brain network using differential equations on a graph Laplacian derived from connectome data.

### 🔍 CLI-Enabled Scripts

All major scripts support command-line arguments for reproducibility and modularity.

For example:

```bash
python src/analysis/braak5_threshold.py --threshold 0.2 --plot
python experiments/full_model_comparison.py --save
```

---

## ▶️ Example Usage

```bash
# Compare all four models
python experiments/full_model_comparison.py

# Show Laplacian evolution over time
python src/visualisation/laplacian_heatmaps_plot.py

# Analyse when Braak V activates under FKPP + clearance
python src/analysis/braak5_threshold.py --threshold 0.15 --plot

# Plot regime differences
python src/analysis/regimes_plot.py
```

---

## 📊 Visualisations

- **Braak activation curves** (stages I–VI + rest of brain)
- **Tau biomarker accumulation** over time
- **Laplacian heatmaps** for dynamically evolving graphs
- **Interactive brain network** colored by Braak zones (via Plotly)

---

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone git@github.com:IsmaLeal/alzheimers-progression.git
   cd alzheimers-progression
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📁 Data

CSV files under `databases/` provide:

- Structural connectome matrices (A0, A1, A2)
- Brain region volumes
- 3D coordinates for plotting
- Braak staging node definitions

These were preprocessed and loaded automatically in `src/data/preprocessing.py`.

---

## 💡 Credits and Notes

- Scientific basis: Prion-like propagation, FKPP PDEs, Braak staging
- Developed as part of a mathematical modelling case study
- All visual styles follow a clean serif + LaTeX look, tuned for publication-ready plots

---

## 🧼 To Do / Ideas

- Add parameter sweeps for robustness testing
- Animate Laplacian evolution
- Export plots directly to PDF
- Expand to include PET scan comparisons
