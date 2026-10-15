# The Greenlight Engine: ML Data Architecture 🎬

## The Vision
In the real world, data doesn't arrive in clean matrices—it is a chaotic mix of text, numbers, and missing files. The Greenlight Engine is a Phase 2 Machine Learning project that demonstrates how to build an indestructible data processing assembly line. It ingests messy studio data, repairs it, translates it into pure mathematics, and uses Logistic Regression to predict a binary outcome: Hit (1) or Flop (0).

## The Architecture: The Split Factory
To prevent mathematical engines from crashing on text or missing data, this pipeline utilizes a Scikit-Learn `ColumnTransformer` to route data through specialized preprocessors before hitting the final engine.

1. **The Numeric Pipeline:** 
   - Captures continuous numbers (e.g., `Budget`, `Runtime`).
   - Uses `SimpleImputer(strategy='mean')` to mathematically patch missing data (`NaN`).
   - Uses `StandardScaler()` to shrink massive numbers into Z-scores so variables don't overpower each other.

2. **The Categorical Pipeline:**
   - Captures text strings (e.g., `Genre`).
   - Uses `SimpleImputer(strategy='most_frequent')` to fill blank categories.
   - Uses `OneHotEncoder()` to destroy the text and convert it into a sparse binary matrix (1s and 0s) to prevent hierarchical bias.

3. **The Master Merger & Engine:**
   - The `ColumnTransformer` stitches the clean numbers and one-hot encoded text back together.
   - The finalized matrix is fed into a `LogisticRegression` classifier for the final prediction.

## Tech Stack
* **Python 3**
* **NumPy & Pandas:** Data manipulation and matrix structuring.
* **Scikit-Learn:** Pipeline architecture (`Pipeline`, `ColumnTransformer`), Preprocessing (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`), and Machine Learning (`LogisticRegression`).
* **Evaluation:** Advanced model grading using `confusion_matrix` (The 4 Rooms) and `classification_report` (Precision, Recall, F1-Score).
