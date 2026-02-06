# Spam-Analysis

This project is an end-to-end Spam Detection System that classifies emails/messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and a Bidirectional LSTM neural network built with TensorFlow/Keras.

🚀 Project Overview

Spam messages waste time and can be dangerous. This project automatically detects whether a given email/message is spam or not spam using:

✔ NLP text preprocessing

✔ One-hot encoding + padding

✔ Deep Learning with Bi-LSTM

✔ Binary classification using Sigmoid

🧠 What I Have Done in This Project

Loaded and explored the dataset

Cleaned and preprocessed the text

Converted text into numerical form

Built a deep learning model

Trained and validated the model

Tested performance

Saved the trained model

Loaded the model in PyCharm

Created a frontend UI for user input

📂 Dataset Description

The dataset file is: spam.csv

Original columns:

v1 → Label (ham / spam)

v2 → Email / Message text

Dataset Processing:

• Dropped unused columns

• Renamed:

v1 → label

v2 → Mails
• Mapped:

ham → 1

spam → 0

🧹 Data Cleaning (NLP)

Each email is processed using:

✔ Lowercasing

✔ Removing punctuation

✔ Removing stopwords

✔ Lemmatization

Example:

"WIN a FREE ticket now!!!"

→ "win free ticket"

🔢 Text Vectorization

• Used one_hot() encoding

• Vocabulary size = 5500

• Converted each sentence into a list of integers

• Used pad_sequences() so all inputs have the same length

✂ Data Splitting

Data	Samples

Training -> First 5000

Validation -> 500

Testing	-> Remaining

🏗 Model Architecture (Bi-LSTM)

Embedding Layer

→ Masking Layer

→ Bi-LSTM (3 units)

→ Bi-LSTM (4 units)

→ Bi-LSTM (5 units)

→ Dense (Sigmoid Output)


Why Bi-LSTM?

✔ Reads text forward and backward

✔ Captures context better

⚙ Model Training

• Optimizer: Adam

• Loss: Binary Crossentropy

• Metric: Accuracy

• Epochs: 20

• Batch Size: 50

💾 Model Saving

The trained model is saved as:

spam_review.pkl

🔮 Prediction: How to Use the Model

After training, the model can predict whether a new message is spam or not.

Step 1: Load the Saved Model

import pickle

with open('spam_review.pkl', 'rb') as f:

    model = pickle.load(f)

Step 2: Preprocess New Input Text

You must apply the same cleaning steps used in training:

✔ Lowercase

✔ Remove punctuation

✔ Remove stopwords

✔ Lemmatize

✔ One-hot encode

✔ Pad sequence

Step 3: Predict

result = model.predict(input_data)

if result > 0.5:
    print("Ham")
else:
    print("Spam")

🖥 Frontend + PyCharm Integration

After training the model in Colab :

✔ I loaded the saved model in PyCharm

✔ Created a frontend UI where users can:

Enter a message

Click Predict

Get Spam / Ham output

The frontend connects to the model and sends user input for prediction in real-time.

Features:

• Clean UI for user input

• Real-time spam detection

• Works as a mini web app / desktop app

📌 Future Improvements

✔ Add Email API Integration

✔ Use Word2Vec / GloVe / BERT

✔ Improve UI design

✔ Deploy on cloud

👩‍💻 Author

Siva Sai Sravani

Data Science & ML Enthusiast

Email: sivasaisravani@gmail.com

LinkedIn: https://www.linkedin.com/in/siva-sai-sravani/

GitHub: https://github.com/sravanibatta15
