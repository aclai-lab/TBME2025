# Eye-Tracking Data Cleaning, Processing, Learning & Knowledge Distillation

This repository contains two Jupyter notebooks and a `requirements.txt` file. The notebooks demonstrate a complete pipeline for processing eye-tracking data, extracting relevant features, and performing a classification task with feature selection. Below is an overview of the data structure, the workflow, and the instructions to reproduce the results.

---

## Repository Contents

1. **`DataCleaning_Pipeline.ipynb`**  
   - Performs data cleaning, merging of pupil coordinates with pupil diameter and confidence values, and extracts relevant features (fixations, areas of interest, etc.) for each trial.
   - Outputs a final dataset containing trials and their extracted features, along with the corresponding *vote* (i.e., user rating).

2. **`Learning_KnowledgeDistillation.ipynb`**  
   - Executes the machine learning experiments discussed in the accompanying paper:
     - Uses [PyGAD](https://pygad.readthedocs.io) for feature selection and 10-fold cross-validation of a Random Forest classifier.
     - Evaluates model performance in terms of accuracy, sensitivity, specificity, PPV, and NPV.
     - Compares different time-window splits (finestrations) of the data.
     - Chooses the best feature subset that yields the highest accuracy for a binary classification of the vote.

3. **`requirements.txt`**  
   - Contains the Python dependencies necessary to run the notebooks.

---

## Data Description

### Input Files

Each trial is defined by a pair (*subject*, *painting*). For each trial, you will have:

1. **Pupil Coordinates Files (`trial_i.csv`)**  
      - Contains the *x*, *y* coordinates of the pupil at each timestamp.
      - Each row corresponds to a single timestamp.

2. **Diameter & Confidence Files**  
      - Contains the pupil diameter and a confidence score for each timestamp (extracted from the same trial).
      - Each row corresponds to a single timestamp, in sync with the coordinates file.

3. **`votes.csv`**  
      - Associates each (*subject*, *painting*) pair with the *vote* (a rating) the subject assigned to that artwork.

Below is an illustrative example of how the merged data (timestamp-wise) might look once coordinates, diameter, and confidence are combined:

| timestamp | x      | y      | diameter | confidence |
|-----------|--------|--------|----------|------------|
| 1         | 12.35  | 34.56  | 5.23     | 0.98       |
| 2         | 14.65  | 36.78  | 5.12     | 0.95       |
| 3         | 13.80  | 35.10  | 5.32     | 0.99       |
| 4         | 12.90  | 36.20  | 5.07     | 0.94       |

### Trial Duration

   Knowing the sampling frequency (in Hz) allows you to convert the number of rows (timestamps) into the duration (in seconds) of each trial.

---

## Notebook 1: `DataCleaning_Pipeline.ipynb`

### Overview

1. **Data Merging**  
      Merges the coordinates file(s) with the diameter/confidence file(s) for each trial. The result is a complete dataset of (*timestamp*, x, y, *diameter*, *confidence*) for each trial.

2. **Data Cleaning**  
      - Removes rows/trials based on statistical dispersion thresholds (both in terms of duration and range of values).
      - Ensures high-quality measurements remain in the final dataset.

3. **Fixation Detection**  
      - Dynamically calculates a threshold for each trial (based on the dispersion of pupil coordinates).
      - Identifies fixations (clusters of consecutive samples where the pupil position is relatively stable) and assigns them an area of interest.

4. **Feature Extraction**  
      - Extracts various features for each fixation and for the entire trial:
        - **Spatial features** (e.g., centroid of fixations, bounding boxes).
        - **Temporal features** (e.g., duration of fixations, time to first fixation).
        - **Pupil-based features** (e.g., average diameter, standard deviation of diameter).
      - Aggregates these features in multiple ways (e.g., summary statistics per fixation, or across all fixations in a trial).

5. **Dataset Construction**  
      - The final step merges all features for each (*subject*, *painting*) trial with the corresponding *vote* from `votes.csv`.
      - Outputs a “ready-to-use” dataset for machine learning models.

N.B. After the cleaning pipeline it will be occupied nearly 500MB of residual files of intermediate steps.

### How to Run

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/aclai-lab/TBME2025.git TBME2025/
   cd TBME2025
   ```

2. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

3. **Place Your Data**

    Organize your trial CSV files (coordinates and diameter/confidence) in the expected directory structure (see the notebook for the exact folder paths).
    Ensure votes.csv is present and correctly formatted.

4. **Launch Jupyter Notebook**

    ```bash
    jupyter notebook
    ```

Open DataCleaning_Pipeline.ipynb and run the cells in sequence.
Verify that the final output is the cleaned dataset (saved as a CSV or Pandas DataFrame, depending on your configuration).

## Notebook 2: Learning_KnowledgeDistillation.ipynb

### Overview

1. **Feature Selection & Genetic Algorithm**
      Uses PyGAD to evolve subsets of features.
      Each generation attempts to maximize the classification accuracy of a Random Forest in a 10-fold cross-validation setup.

2. **Multiple Time-Window Splits**
      The dataset can be split into up to four different time windows (or “finestrations”).
      The objective is to find which segment of the trial yields the highest predictive accuracy.

3. **Random Forest Training**
        Trains a Random Forest in each generation with the selected subset of features.
        Evaluates the model using:
            Accuracy
            Sensitivity (True Positive Rate)
            Specificity (True Negative Rate)
            PPV (Positive Predictive Value)
            NPV (Negative Predictive Value)
        Repeats the evaluation over 10 folds to produce an average and standard deviation.

4. **Best Subset & Final Models**
        After 100 generations, identifies the best performing feature subset (the highest accuracy across folds).
        Trains and evaluates 10 Random Forest models using that best feature subset.
        Selects the final model (and feature set) that achieves the highest overall accuracy.

### How to Run

  1. **Cleaned Dataset**
        Ensure you have run DataCleaning_Pipeline.ipynb and generated the final dataset with features and votes.

  2. **Install Dependencies**
        Already covered by requirements.txt (particularly PyGAD and scikit-learn).

  3. **Open the Notebook**
        Run jupyter notebook from the repository root.
        Open Learning_KnowledgeDistillation.ipynb.

  4. **Set the Hyperparameters**
        Adjust PyGAD parameters (e.g., number of generations, population size, crossover/mutation rates) if needed.
        Adjust Random Forest hyperparameters as desired.

  5. **Execute the Cells**
        The notebook will:
            Split the dataset into multiple time windows.
            Run a 10-fold cross-validation within each generation of PyGAD.
            Print out the best accuracy, sensitivity, specificity, PPV, NPV, and standard deviations across folds.
            Display the best feature subset found and the final performance metrics.

## Results Summary

  **Optimal Time-Window**
     A specific window (or combination of windows) produced the highest accuracy. For example, [0,11] -> 0 and [42,50] -> 1 in the best-performing scenario.

  **Highest Accuracy**
     After 100 generations, we observe improved accuracy, sensitivity, specificity, PPV, and NPV compared to the baseline model trained on the entire feature set.

  **Feature Subset**
     The final model used a carefully selected subset of features, which were found to maximize classification performance in a 10-fold cross-validation setting.

   Reproducing the Experiment
   Run DataCleaning_Pipeline.ipynb to generate the cleaned dataset.
   Run Learning_KnowledgeDistillation.ipynb to perform the feature selection, train the models, and evaluate their performance.
   Interpret the Results:
      Check the printed metrics and generation logs in the notebook outputs.
      Refer to the final “best subset” of features for subsequent analyses or real-world applications.

### Contributing

   Feel free to open issues or submit pull requests. If you have ideas to improve the data cleaning or the feature selection strategy (e.g., integrating domain-specific insights about eye movements and psychology), please contribute!

### License

   This project is licensed under the MIT License – see the LICENSE file for details.

### Contact

   For questions about this repository or the associated paper, please open an issue or reach out directly. Contributions and collaborations are welcome.

