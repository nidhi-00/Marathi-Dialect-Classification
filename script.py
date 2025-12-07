# Step 1: Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load the dataset
train_df = pd.read_csv("train_data.csv")   # adjust path if needed
test_df  = pd.read_csv("test_data.csv")    # adjust path if needed

# Inspect the first few rows to verify the data format
print("Training data sample:")
print(train_df.head(5))
print("\nNumber of training samples:", len(train_df))
print("Number of testing samples:", len(test_df))
print("Dialect labels in training data:", train_df['label'].unique())

# Step 3: Prepare features using TF-IDF vectorization

# Separate features and labels for convenience
X_train_texts = train_df['sentence'].astype(str)  # ensure text is string
y_train = train_df['label']
X_test_texts  = test_df['sentence'].astype(str)
y_test  = test_df['label']

# Initialize a TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 1),  # using unigrams; you can try (1,2) for including bigrams
                             lowercase=False)     # lowercase=False since Devanagari has no case

# Learn vocabulary from training texts and transform training data into feature vectors
X_train_vec = vectorizer.fit_transform(X_train_texts)

# Transform the test data into feature vectors using the same vocabulary
X_test_vec = vectorizer.transform(X_test_texts)

print("TF-IDF vectorization complete.")
print("Number of features (vocabulary size):", X_train_vec.shape[1])

# Step 4: Train the Logistic Regression model

# Initialize the logistic regression model
model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)  
# multi_class 'auto' lets sklearn choose one-vs-rest or multinomial based on solver
# solver 'lbfgs' is efficient for multiclass, increase max_iter if needed.

# Train the model on the TF-IDF features and corresponding labels
model.fit(X_train_vec, y_train)

print("Model training complete.")

# Step 5: Evaluate the model on the test set

# Predict dialect labels for the test set
y_pred = model.predict(X_test_vec)

# Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Display a classification report for detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Optionally, display a confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['D1','D2','D3','D4'])
print("Confusion Matrix (rows=true, cols=pred):\n", cm)

# Step 6: Using the model to predict new examples
new_sentences = [
    "तो आज गावाहून शहराला गेला.",    # Replace with any Marathi sentence in Devanagari
    "तू काय करतोयस?",              # Another example sentence
]
# Convert new sentences to TF-IDF vectors (using the fitted vectorizer)
new_vec = vectorizer.transform(new_sentences)
# Predict using the trained model
predictions = model.predict(new_vec)

for sent, pred in zip(new_sentences, predictions):
    print(f"Sentence: {sent} --> Predicted Dialect: {pred}")


