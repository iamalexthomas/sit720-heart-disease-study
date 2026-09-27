# Heart disease classification study

**SIT720 Machine Learning — Alex Thomas, 225706598**

This assignment reproduces the models in *An efficient stacking-based ensemble
technique for early heart attack prediction* by Bhagat, Sharma and Agarwal
([paper DOI](https://doi.org/10.1007/s11042-024-19293-7)). It then compares the
stacking model with a simpler logistic regression model.

Start with the [notebook](research_notebook.ipynb) for the explanation and outputs,
or read the [PDF report](report/SIT720_HD_Report.pdf).
An editable [Word report](report/SIT720_HD_Report.docx) is also included.

## Main steps

1. Check the data and reproduce the six individual models and stacking model.
2. Remove duplicate records before splitting the data.
3. Scale numeric features and one-hot encode categorical features.
4. Fit logistic regression and choose a threshold using training folds.
5. Compare the models on the same ten train/test splits.

There are 1,025 rows but only 302 distinct records. On the original split,
199 of 205 test rows have a copy in training. After removing duplicates, the
proposed method has higher average recall than stacking, but lower precision
and slightly lower accuracy. The report discusses these trade-offs.

## Run the code

The experiments were tested with Python 3.14.7 on Linux. The models run on a CPU
and use the included CSV; no data download is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiments.py
python tools/verify_results.py
```

On Windows, use `python` instead of `python3` and activate the environment with
`.venv\Scripts\activate`. Installing the packages needs internet. The exact
supporting versions used for the experiments are in `tools/requirements-lock.txt`.

To run the notebook, select the same virtual environment as its Python kernel,
then choose **Restart Kernel and Run All**. The saved notebook already contains
executed outputs. It calls the same experiment script and explains the main steps.

## Folder guide

| File or folder | Contents |
| --- | --- |
| `research_notebook.ipynb` | Step-by-step explanation and executed outputs |
| `run_experiments.py` | Data checks, models, comparisons and plots |
| `data/` | Original CSV and its source information |
| `results/` | Saved scores, predictions, splits and figures |
| `report/` | PDF, Word report, editable text and video guide |
| `tools/` | Helpers for checking results and rebuilding the report |

The experiment code uses ordinary functions, pandas and scikit-learn pipelines.
The extra files in `results/` allow the reported scores to be checked without
retraining. `record_id` refers to the original CSV row number, not a patient ID.
Class 1 is the supplied positive label; this study does not establish future
heart attack prediction or clinical suitability.

## Update the report

The video link is left blank. After recording, add it to
`report/submission_details.json`, then run:

```bash
python tools/build_report.py
python tools/package_submission.py
```

The first command rebuilds the Markdown, PDF and Word reports from the saved
results. To change the report wording permanently, edit the text in
`tools/build_report.py`; rebuilding replaces `report/report.md`.
The second command creates a ZIP beside the project folder.

[Video guide](report/VIDEO_GUIDE.md) · [Study plan](report/study_plan.md) ·
[Data source](data/SOURCE.md)
