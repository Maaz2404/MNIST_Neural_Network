import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.title("MNIST Neural Net")

def load_data():
    folder_path = 'dataframes'
    with open(f'{folder_path}/trainx.sav', 'rb') as f:
        trainx = pickle.load(f)
        trainx = tf.keras.utils.normalize(trainx, axis=1)
    with open(f'{folder_path}/trainy.sav', 'rb') as f:
        trainy = pickle.load(f)
        
    with open(f'{folder_path}/devx.sav', 'rb') as f:
        devx = pickle.load(f)
        devx = tf.keras.utils.normalize(devx, axis=1)
    with open(f'{folder_path}/devy.sav', 'rb') as f:
        devy = pickle.load(f)
        
    return trainx, trainy, devx, devy

trainx, trainy, devx, devy = load_data()

def build_nn(hidden_layers):
    units_per_layer = []
    for i in range(hidden_layers):
        units = st.sidebar.slider(f"Units in hidden layer {i+1}", min_value=1, max_value=128, value=32, step=1)
        units_per_layer.append(units)
        st.write(f"Layer {i+1}: {units} units")
    return units_per_layer

st.sidebar.header("Configure Neural Network")
hidden_layers = st.sidebar.slider("Number of hidden layers", min_value=1, max_value=5, value=1, step=1)
nn_structure = build_nn(hidden_layers)
epochs = st.sidebar.slider("Select number of epochs", min_value=1, max_value=50, step=5)

def create_model(layers):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(784,)))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if st.sidebar.button("TRAIN"):
    model = create_model(nn_structure)
    history = model.fit(trainx, trainy, epochs=epochs, validation_data=(devx, devy), verbose=1)
    model.save("neural_net.keras")

    st.subheader("Training Loss and Accuracy")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    st.write("Model training completed.")

if st.sidebar.button("EVALUATE RESULTS"):
    # if not tf.io.gfile.exists("neural_net.keras"):
    #     st.error("No trained model found. Please train the model first.")
    # else:
        model = tf.keras.models.load_model("neural_net.keras")
        human_acc = 0.99
        st.write("Human-level accuracy: 0.99")
        # Evaluate on training set
        train_loss, train_accuracy = model.evaluate(trainx, trainy, verbose=0)
        st.write(f"Training Loss: {train_loss:.4f}")
        st.write(f"Training Accuracy: {train_accuracy:.4f}")

        # Evaluate on validation (dev) set
        dev_loss, dev_accuracy = model.evaluate(devx, devy, verbose=0)
        st.write(f"Validation Loss: {dev_loss:.4f}")
        st.write(f"Validation Accuracy: {dev_accuracy:.4f}")

        # Determine model status
        if (human_acc - train_accuracy) >= 0.03 :
            st.write("The model might be underfitting. Consider increasing model complexity.")
        elif (train_accuracy - dev_accuracy) >=0.03:
            st.write("The model might be overfitting. Consider using regularization or adding more data.")
        else:
            st.write("The model seems to be performing well on both training and validation sets.")
