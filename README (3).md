# Phishing Email Detection Model

A beginner-friendly machine-learning project that classifies email text as **Phishing** or **Safe** using Python and Scikit-learn.

## Features
- Reads labeled emails from a CSV dataset.
- Uses TF-IDF to extract text features.
- Counts URLs and suspicious keywords and measures email length.
- Trains a Logistic Regression classifier.
- Displays accuracy, classification report, and confusion matrix.
- Saves the confusion matrix and trained model.
- Lets you test a new email message.

## Files
- `phishing_email_detector.py` — main Python program.
- `phishing_emails.csv` — small sample dataset to test the project.
- `README.md` — project instructions.

## 1. Install required libraries
Open Command Prompt in this folder and run:

```bash
py -m pip install pandas scikit-learn matplotlib scipy joblib
```

If `py` does not work, try `python` instead.

## 2. Dataset format
The CSV must contain two columns: `email_text` and `label`.

Accepted labels: `Phishing` and `Safe`, `spam` and `ham`, or `1` and `0`.

Example:

```csv
email_text,label
"Urgent: verify your bank account immediately",Phishing
"Click here to claim your prize now",Phishing
"Your class timetable is attached for tomorrow",Safe
"Meeting starts at 10 AM",Safe
```

The included CSV is only a tiny starter dataset to check that the program runs. For meaningful evaluation, replace it with a larger, correctly labeled dataset that you are permitted to use.

## 3. Run the program

```bash
py phishing_email_detector.py
```

The program creates:
- `confusion_matrix.png`
- `phishing_email_model.joblib`

It also asks you to enter an email for a sample prediction.

## 4. How it works
1. Loads and validates the dataset.
2. Splits raw emails into training and test data.
3. Fits TF-IDF and the numeric-feature scaler using training data only.
4. Combines text and numeric features.
5. Trains Logistic Regression.
6. Evaluates predictions on the test data.
7. Saves the confusion matrix and trained model.

## 5. Understanding results
- **Accuracy:** percentage of test emails classified correctly.
- **Precision:** how many emails predicted as a class were correct.
- **Recall:** how many actual emails in a class were found.
- **F1-score:** combined measure of precision and recall.
- **Confusion matrix:** shows correct and incorrect predictions.

Accuracy depends on the dataset. Do not claim a specific accuracy before running the model on your own dataset.

## Limitations and responsible use
- This is an educational text-based detector, not a complete email security product.
- It does not inspect sender headers, SPF/DKIM/DMARC, attachments, or website contents.
- URLs are counted but never opened.
- A `Safe` prediction is not a guarantee that an email is safe.
- Do not upload private or confidential emails to a public GitHub repository. Use data you have permission to process.

## Technologies
Python, Pandas, Scikit-learn, TF-IDF, Logistic Regression, Matplotlib, SciPy, Joblib.
