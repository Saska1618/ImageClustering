# Exploratory Data Analysis (EDA) for Image Clustering

This document summarizes the EDA steps performed in the `initial_data.ipynb` and `eda.ipynb` notebooks.

## 1. Overview of the Dataset
- **File Used**: `classes.csv`
- **Initial Steps**:
  - Loaded the dataset using `pandas`.
  - Inspected the dataset using methods like `.head()`, `.dtypes`, `.describe()`, and `.info()`.

## 2. Data Cleaning
- Checked for missing values using `.dropna()` and confirmed there were no NaN values.
- Converted string representations of lists in the `genre` column to actual Python lists using `ast.literal_eval`.

## 3. Genre Analysis
- **Genre Distribution**:
  - Counted occurrences of each genre using `collections.Counter`.
  - Visualized the distribution of genres using a horizontal bar chart (`matplotlib`).

## 4. Artist Analysis
- **Artist Frequency**:
  - Counted occurrences of each artist using `.value_counts()`.
  - Visualized the top 10 artists by count using a bar chart (`seaborn`).

## 5. Description Analysis
- Counted occurrences of unique descriptions using `.value_counts()`.

## 6. Cross-tabulation (Optional)
- Created a pivot table to analyze the distribution of genres across artists using `pd.crosstab`.
- Visualized the cross-tabulation for the top 10 artists using a heatmap (`seaborn`).

## 7. Visualizations
- **Tools Used**:
  - `matplotlib` for basic plots.
  - `seaborn` for enhanced visualizations.
- **Key Plots**:
  - Horizontal bar chart for genre distribution.
  - Bar chart for top artists by count.
  - Heatmap for artist-genre cross-tabulation.

## 8. Notes
- The `eda.ipynb` notebook focuses more on visualizations, while `initial_data.ipynb` includes additional data inspection steps.
- Ensure the `genre` column is properly converted to lists before performing any analysis.

## 9. EDA Conclusions
- The dataset contains a diverse set of genres, with some genres being significantly more frequent than others.
- A small number of artists contribute to a large portion of the dataset, as evidenced by the top 10 artists dominating the distribution.
- The `description` column has a high degree of repetition, indicating that many artworks share similar or identical descriptions.
- The cross-tabulation analysis revealed that certain artists specialize in specific genres, while others contribute to a broader range of genres.
- The dataset is clean, with no missing values, and the `genre` column was successfully converted to a usable format for analysis.

These insights will guide the preprocessing and clustering steps in subsequent stages of the project.

## How to Run
1. Place the `classes.csv` file in the appropriate directory.
2. Open the notebooks (`initial_data.ipynb` or `eda.ipynb`) in Jupyter Notebook or Jupyter Lab.
3. Execute the cells sequentially to reproduce the analysis and visualizations.

