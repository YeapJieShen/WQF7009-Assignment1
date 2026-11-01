# WQF7009-Assignment1
WQF7009 Explainable Artificial Intelligence Assignment 1: Using SHAP for Stroke Prediction Model

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # add extras as needed, or pip install -e ".[all]" for everything
```

Download the [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
and place `healthcare-dataset-stroke-data.csv` in `data/` (see [data/README.md](data/README.md)).

## Credits

The starter scripts (`00_`–`05_*.py`, `utils.py`) are adapted from
[deepfindr/xai-series](https://github.com/deepfindr/xai-series).
