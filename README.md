# Heart disease classification study

**SIT720 Machine Learning — Alex Thomas, 225706598**

This assignment reproduces the models in *An efficient stacking-based ensemble
technique for early heart attack prediction* by Bhagat, Sharma and Agarwal
([paper DOI](https://doi.org/10.1007/s11042-024-19293-7)). It then compares the
stacking model with a simpler logistic regression model.

Start with the [notebook](research_notebook.ipynb) for the explanation and outputs.
Both the reproduced experiment and the proposed solution are implemented in
`run_experiments.py`.

## Main steps

1. Check the data and reproduce the six individual models and stacking model.
2. Remove duplicate records before splitting the data.
3. Scale numeric features and one-hot encode categorical features.
4. Fit logistic regression and choose a threshold using training folds.
5. Compare the models on the same ten train/test splits.

There are 1,025 rows but only 302 distinct records. On the original split,
199 of 205 test rows have a copy in training. After removing duplicates, the
proposed method has higher average recall than stacking, but lower precision
and slightly lower accuracy. The notebook discusses these trade-offs.

## Run the code

The experiments were tested with Python 3.14.7 on Linux. The models run on a CPU
and use the included CSV; no data download is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sit720-heart-study --display-name "Python (SIT720 project)"
python run_experiments.py
python tools/verify_results.py
```

On Windows, use `python` instead of `python3` and activate the environment with
`.venv\Scripts\activate`. Installing the packages needs internet. The exact
supporting versions used for the experiments are in `tools/requirements-lock.txt`.

To run the notebook, select **Python (SIT720 project)** as its kernel. In VS Code,
click the kernel name at the top right, then **Select Another Kernel → Jupyter
Kernels → Python (SIT720 project)**. In Jupyter, use **Kernel → Change Kernel**.
If the new kernel does not appear immediately, reopen the notebook application.

For the full study, choose **Restart Kernel and Run All**. The first code cell
checks the environment and finds the project folder. The saved notebook already
contains executed outputs.

For the video, you can run **Section 6: Live demonstration for the video** on its
own after restarting the kernel. It loads its own data and imports, so you do not
need to rerun all the experiments during the recording.

If you see `No module named 'sklearn'`, check the selected kernel: the system
Python may not have the project packages. The package installed by pip is named
`scikit-learn`; its Python import name is `sklearn`.

## Folder guide

| File or folder | Contents |
| --- | --- |
| `research_notebook.ipynb` | Step-by-step explanation and executed outputs |
| `run_experiments.py` | Data checks, models, comparisons and plots |
| `data/` | Original CSV and its source information |
| `results/` | Saved scores, predictions, splits and figures |
| `tools/verify_results.py` | Checks the saved scores against the predictions |
| `tools/requirements-lock.txt` | Exact supporting package versions |

The experiment code uses ordinary functions, pandas and scikit-learn pipelines.
The extra files in `results/` allow the reported scores to be checked without
retraining. `record_id` refers to the original CSV row number, not a patient ID.
Class 1 is the supplied positive label; this study does not establish future
heart attack prediction or clinical suitability.

[Data source](data/SOURCE.md)
