# Recording guide - aim for about 5 minutes 30 seconds

This is a script to adapt after you understand the work. It is not a finished
video and should not be presented as evidence of steps you have not personally
checked. Use your own words and add a truthful reflection on what you learned.

The task requires your **face**, **screen sharing**, and **clear voice narration**.
The maximum length is six minutes. Keep your webcam visible while showing the
executed notebook. Rehearse once and shorten explanations if you go over time.

## Before recording

Open `research_notebook.ipynb` and select the project environment. Run it once
so the outputs are visible. Keep the source file, comparison table and confusion
matrices ready. Increase the editor font so viewers can read the selected code.
Use the notebook's small live-demo cell during recording; you do not need to
wait for the entire ten-split experiment on camera.

## 0:00-0:40 - introduce the paper

Show the report title and the paper citation.

"This project reproduces paper option three, which combines six models using
stacking. It uses a small heart-disease dataset with 13 input measurements and
a binary target. I chose it because standard Python libraries can implement
the models, and the data is easier to work with than raw ECG recordings.
The main question is whether the high accuracy holds up when repeated records
are kept out of the test set."

## 0:40-1:25 - explain the reproduction

Show `make_models` and the Part 1 table.

"The six models are logistic regression, decision tree, random forest, XGBoost,
Naive Bayes and KNN. Stacking combines their predicted probabilities using a
final logistic regression model. Five-fold predictions from the training set
teach the final model how to combine them. Each base learner has its own scaler.
The paper does not state every setting, so the report lists assumptions,
including random seed 42 and the final classifier. The 80/20 split is inferred
from the paper's 205 test observations."

## 1:25-2:10 - explain the important finding

Show the duplicate audit, then the first accuracy table.

"The raw-split stack reaches 98.54 percent accuracy, close to the paper's
98.53 percent. But the dataset has 1,025 rows and only 302 distinct records.
In this split, 199 of the 205 test rows have an identical record in training.
That makes the score less convincing as a measure of performance on unseen
records. It does not prove that the authors used the same split. The dataset
also does not establish prediction of a future heart attack."

## 2:10-3:10 - explain and demonstrate the proposal

Show the proposed model and run the notebook cell marked **Live demonstration**.
Point to the printed threshold and evaluation metrics.

"The alternative removes duplicate records, scales the continuous measurements,
and turns category codes into separate yes-or-no columns. It then uses one
logistic regression model. I also test a threshold chosen from training-fold
predictions to favour recall. The F2 score gives missed positive records more
weight than false positives. The test labels are not used to select that
threshold. This changes the whole data and decision pipeline, not just the
classifier name. This demo uses seed 42, with 241 training and 61 test records."

## 3:10-4:15 - compare the results fairly

Show the ten-split table and comparison figure.

"All seven original methods are also retrained on the same duplicate-free
splits. The proposed method has mean recall of 92.42 percent, compared with
86.67 percent for stacking. AUC improves from about 0.904 to 0.922. However,
accuracy is 81.97 percent, compared with 82.95 percent for stacking, and
precision is lower. The one-hot model at the usual 0.5 threshold actually has
the best mean accuracy, 85.90 percent. So the threshold is useful for the
recall objective, but it is not best for every purpose."

## 4:15-5:00 - show a failure case

Show the three confusion matrices.

"On the seed-42 test set, the stack misses seven positive records and the
proposal misses five. However, compared with one-hot logistic regression at
0.5, threshold tuning adds a false positive without finding another positive.
This is an example where the extra tuning does not help. The error bars show
variation across ten splits, not a confidence interval. The same records
reappear across splits, so these are not ten independent datasets."

## 5:00-5:35 - reflect and conclude

Show the report's limitations.

"The main lesson from these experiments is to inspect the data split before
trusting a very high accuracy. A simpler model can be competitive after the
evaluation is made more careful. There are only 302 distinct records, no patient
IDs, and no external test dataset. Next steps would be to verify the label
meanings and test on independent records. The report includes the model assumptions and the limitations of these results."

Add one short, truthful sentence about a concept you found difficult and how
you now understand it. Do not claim this system can diagnose patients.

## After recording

Upload the video as YouTube Unlisted or to another accessible platform. Add its
link to `report/submission_details.json` and run `python tools/build_report.py`.
The GitHub link is already included. Replace the reports in the repository, check
both links while signed out, and submit the PDF through OnTrack.

## Quick understanding check

- Why can duplicate rows make a test result misleading?
- What is the difference between a base classifier and a stacking meta-classifier?
- Why must the scaler and encoder be fitted inside training folds?
- How can lowering a threshold raise recall and lower precision?
- Why does changing only the threshold leave AUC unchanged?
- Why is comparing two models on identical test records more convincing?
- Which settings came from the paper, and which were assumptions?
