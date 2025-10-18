import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64

# Load Data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

# Pad sequences
train_data = sequence.pad_sequences(train_data, maxlen=MAXLEN)
test_data = sequence.pad_sequences(test_data, maxlen=MAXLEN)

# Build Model
model = Sequential([
    Embedding(VOCAB_SIZE, 32),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

# Compile Model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2, batch_size=BATCH_SIZE)

# Evaluate
results = model.evaluate(test_data, test_labels)
print(results)

#saving model for later loading
model.save("model.h5")
#testing
index=[0]

test_data = sequence.pad_sequences(test_data, maxlen=MAXLEN)

prediction= model.predict(test_data)

print("True Label:", test_labels[index])  # 0 = negative, 1 = positive
value = prediction[0][0]   # or float(prediction)
if value > 0.5:
    print("Positive")
else:
    print("Negative")

print("confidence score:",prediction[0][0])
