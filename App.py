
import streamlit as st
import numpy as np
from keras.models import load_model

st.title('Stock Price Prediction App')

# Load the trained Keras model
model = load_model('C:/Projects/Sales Forecasting/simple_rnn_model.keras')

st.write("Model 'simple_rnn_model.keras' loaded successfully!")


st.subheader('Enter 10 Historical Stock High Prices')

historical_prices = []
for i in range(1, 11):
    price = st.number_input(f'Day {i} High Price', min_value=0.0, format='%.2f', key=f'day_{i}')
    historical_prices.append(price)


if st.button('Predict Next Day High Price'):
    if len(historical_prices) == 10 and all(isinstance(p, (int, float)) for p in historical_prices): # Ensure 10 valid inputs
        # Prepare the input for the model
        input_data = np.array(historical_prices).reshape(1, 10, 1)
        
        # Make prediction
        predicted_price = model.predict(input_data)
        
        # Display the prediction
        st.subheader('Predicted Next Day High Price:')
        st.write(f'{predicted_price[0][0]:.2f}')
    else:
        st.warning('Please enter 10 valid historical high prices.')    