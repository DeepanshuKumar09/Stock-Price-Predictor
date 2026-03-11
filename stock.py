import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, SimpleRNN
from keras.optimizers import Adam
import math
import statistics
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title='Stock Price Predictor', layout='wide')



@st.cache_data
def load_data():
    df = pd.read_csv("C:/Projects/Sales Forecasting/yahoo_stock (1).csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    new_df = df['High'].iloc[:-4]
    return df, new_df

df, new_df = load_data()

st.title('Yahoo Stock Price Predictor')

st.subheader('Raw Data')
st.write(df.head())


st.subheader('Descriptive Statistics')
st.write(df.describe())

st.subheader('Rolling Mean and Standard Deviation of High Prices')
mean = []
std = []
for i in range(0, 10):
    mean.append(df['High'].iloc[(i * 182):(i * 182) + 182].mean())
    std.append(statistics.stdev(df['High'].iloc[(i * 182):(i * 182) + 182]))

st.write(pd.concat([pd.DataFrame(mean, columns=['mean']), pd.DataFrame(std, columns=['std'])], axis=1))

st.subheader('Augmented Dickey-Fuller Test for Stationarity')
result = adfuller(df["High"])
st.write('ADF Statistic: %f' % result[0])
st.write('p-value: %f' % result[1])
st.write("Critical Values:")
for key, value in result[4].items():
    st.write('\t%s: %.3f' % (key, value))


st.subheader('Stock Price Trends')
fig1, ax1 = plt.subplots(figsize=(20, 10))
df[['High', 'Low', 'Open', 'Close']].plot(ax=ax1, alpha=0.5)
ax1.legend(['High', 'Low', 'Open', 'Close'])
ax1.set_title('Stock Price Trends Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
st.pyplot(fig1)

st.subheader('Distribution of High Prices')
fig2, ax2 = plt.subplots()
df['High'].hist(bins=100, ax=ax2)
ax2.set_title('Histogram of High Prices')
ax2.set_xlabel('High Price')
ax2.set_ylabel('Frequency')
st.pyplot(fig2)


st.subheader('Seasonal Decomposition (Additive Model)')
decompose_add = seasonal_decompose(df['High'], model='additive', period=12)
fig_add, axes_add = plt.subplots(4, 1, figsize=(15, 15))

axes_add[0].plot(df['High'], label='Original TS')
axes_add[0].legend(loc='best')
axes_add[0].set_title('Original Time Series')

axes_add[1].plot(decompose_add.trend, label='Trend')
axes_add[1].legend(loc='best')
axes_add[1].set_title('Trend Component')

axes_add[2].plot(decompose_add.seasonal, label='Seasonality')
axes_add[2].legend(loc='best')
axes_add[2].set_title('Seasonal Component')

axes_add[3].plot(decompose_add.resid, label='Residual')
axes_add[3].legend(loc='best')
axes_add[3].set_title('Residual Component')

plt.tight_layout()
st.pyplot(fig_add)

st.subheader('Seasonal Decomposition (Multiplicative Model)')
decompose_mul = seasonal_decompose(df['High'], model='multiplicative', period=12)
fig_mul, axes_mul = plt.subplots(4, 1, figsize=(15, 15))

axes_mul[0].plot(df['High'], label='Original TS')
axes_mul[0].legend(loc='best')
axes_mul[0].set_title('Original Time Series')

axes_mul[1].plot(decompose_mul.trend, label='Trend')
axes_mul[1].legend(loc='best')
axes_mul[1].set_title('Trend Component')

axes_mul[2].plot(decompose_mul.seasonal, label='Seasonality')
axes_mul[2].legend(loc='best')
axes_mul[2].set_title('Seasonal Component')

axes_mul[3].plot(decompose_mul.resid, label='Residual')
axes_mul[3].legend(loc='best')
axes_mul[3].set_title('Residual Component')

plt.tight_layout()
st.pyplot(fig_mul)




from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
st.subheader('Autocorrelation Function (ACF)')
fig_acf, ax_acf = plt.subplots(figsize=(15, 7))
plot_acf(df['High'], ax=ax_acf)
ax_acf.set_title('Autocorrelation Function for High Prices')
st.pyplot(fig_acf)

st.subheader('Partial Autocorrelation Function (PACF)')
fig_pacf, ax_pacf = plt.subplots(figsize=(15, 7))
plot_pacf(df['High'], ax=ax_pacf)
ax_pacf.set_title('Partial Autocorrelation Function for High Prices')
st.pyplot(fig_pacf)



st.subheader('LSTM Model Training and Evaluation (Window=10)')

# 1. Define train_len and window
train_len = math.ceil(len(new_df) * 0.8)
window = 10

st.write(f"Training data length: {train_len}")
st.write(f"Window size for LSTM: {window}")

# 2. Create training data (X_train, Y_train)
train_data = new_df[0:train_len]
X_train = []
Y_train = []

for i in range(window, len(train_data)):
    X_train.append(train_data[i-window:i])
    Y_train.append(train_data[i])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Create validation data (X_val, Y_val)
test_data = new_df[train_len-window:]
X_val = []
Y_val = []

for i in range(window, len(test_data)):
    X_val.append(test_data[i-window:i])
    Y_val.append(test_data[i])

X_val, Y_val = np.array(X_val), np.array(Y_val)
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# 3. Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(25))
model.add(Dense(1))

# 4. Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# 5. Train the model
with st.spinner('Training initial LSTM model...'):
    model.fit(X_train, Y_train, epochs=10, batch_size=10, verbose=0)
st.success('Initial LSTM model trained!')

# 6. Generate predictions
lstm_train_pred = model.predict(X_train, verbose=0)
lstm_valid_pred = model.predict(X_val, verbose=0)

# 7. Calculate and display RMSE
train_rmse = np.sqrt(mean_squared_error(Y_train, lstm_train_pred))
valid_rmse = np.sqrt(mean_squared_error(Y_val, lstm_valid_pred))
st.write(f"Train RMSE: {train_rmse:.2f}")
st.write(f"Validation RMSE: {valid_rmse:.2f}")

# 8. Create valid DataFrame with predictions
valid = pd.DataFrame(new_df[train_len:])
valid['Predictions'] = lstm_valid_pred

st.subheader('Validation Predictions vs. Actuals')
fig_val_pred, ax_val_pred = plt.subplots(figsize=(16, 8))
ax_val_pred.plot(valid[['High', 'Predictions']])
ax_val_pred.legend(['Validation Actuals', 'Predictions'])
ax_val_pred.set_title('LSTM Validation Predictions vs. Actuals')
ax_val_pred.set_xlabel('Date')
ax_val_pred.set_ylabel('High Price')
st.pyplot(fig_val_pred)

# 10. Create combined plot
train = new_df[:train_len]

st.subheader('Train, Validation, and Predictions Plot')
fig_combined, ax_combined = plt.subplots(figsize=(16, 8))
ax_combined.plot(train, label='Train')
ax_combined.plot(valid[['High']], label='Validation Actuals')
ax_combined.plot(valid['Predictions'], label='Predictions')
ax_combined.set_title('LSTM Model: Train, Validation, and Predictions')
ax_combined.set_xlabel('Date')
ax_combined.set_ylabel('High Price')
ax_combined.legend()
st.pyplot(fig_combined)



st.subheader('LSTM Model Performance Across Different Window Sizes')
train_error = []
val_error = []
window_numbers = [5, 8, 10, 15, 20, 30, 40]

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

for idx, current_window in enumerate(window_numbers):
    my_bar.progress((idx + 1) / len(window_numbers), text=f"Processing window size: {current_window}")
    # Prepare data for current window size
    train_data_w = new_df[0:train_len]
    X_train_w = []
    Y_train_w = []
    for i in range(current_window, len(train_data_w)):
        X_train_w.append(train_data_w[i-current_window:i])
        Y_train_w.append(train_data_w[i])

    X_train_w, Y_train_w = np.array(X_train_w), np.array(Y_train_w)
    X_train_w = np.reshape(X_train_w, (X_train_w.shape[0], X_train_w.shape[1], 1))

    test_data_w = new_df[train_len-current_window:]
    X_val_w = []
    Y_val_w = []
    for i in range(current_window, len(test_data_w)):
        X_val_w.append(test_data_w[i-current_window:i])
        Y_val_w.append(test_data_w[i])

    X_val_w, Y_val_w = np.array(X_val_w), np.array(Y_val_w)
    X_val_w = np.reshape(X_val_w, (X_val_w.shape[0], X_val_w.shape[1], 1))

    # Build and train model for current window size
    model_w = Sequential()
    model_w.add(LSTM(50, activation='relu', input_shape=(X_train_w.shape[1], 1)))
    model_w.add(Dense(25))
    model_w.add(Dense(1))
    model_w.compile(loss='mean_squared_error', optimizer='adam')
    model_w.fit(X_train_w, Y_train_w, epochs=10, batch_size=10, verbose=0)

    # Generate predictions and calculate RMSE
    lstm_train_pred_w = model_w.predict(X_train_w, verbose=0)
    lstm_valid_pred_w = model_w.predict(X_val_w, verbose=0)
    train_error.append(np.sqrt(mean_squared_error(Y_train_w, lstm_train_pred_w)))
    val_error.append(np.sqrt(mean_squared_error(Y_val_w, lstm_valid_pred_w)))

st.success('Window size analysis complete!')

results_df = pd.DataFrame({
    'window': window_numbers,
    'train_rmse': train_error,
    'val_rmse': val_error
}).set_index('window')
st.write(results_df)



st.subheader('Complex LSTM Model Evaluation')
window = 10

# Prepare data for new model
train_len = math.ceil(len(new_df) * 0.8)

train_data = new_df[0:train_len]
X_train_complex = []
Y_train_complex = []

for i in range(window, len(train_data)):
    X_train_complex.append(train_data[i-window:i])
    Y_train_complex.append(train_data[i])

X_train_complex, Y_train_complex = np.array(X_train_complex), np.array(Y_train_complex)
X_train_complex = np.reshape(X_train_complex, (X_train_complex.shape[0], X_train_complex.shape[1], 1))

test_data = new_df[train_len-window:]
X_val_complex = []
Y_val_complex = []

for i in range(window, len(test_data)):
    X_val_complex.append(test_data[i-window:i])
    Y_val_complex.append(test_data[i])

X_val_complex, Y_val_complex = np.array(X_val_complex), np.array(Y_val_complex)
X_val_complex = np.reshape(X_val_complex, (X_val_complex.shape[0], X_val_complex.shape[1], 1))


# Define the complex LSTM model
model_complex = Sequential()
model_complex.add(LSTM(50, return_sequences=True, activation='relu', input_shape=(X_train_complex.shape[1], 1)))
model_complex.add(LSTM(50, return_sequences=False, activation='relu'))
model_complex.add(Dense(100))
model_complex.add(Dense(25))
model_complex.add(Dense(1))

# Compile the model with Adam optimizer
opt1 = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model_complex.compile(loss='mean_squared_error', optimizer=opt1)

st.text('Training Complex LSTM Model...')
with st.spinner('Training complex LSTM model (epochs=100)...'):
    model_complex.fit(X_train_complex, Y_train_complex, epochs=100, batch_size=10, verbose=0)
st.success('Complex LSTM model trained!')

# Generate predictions
lstm_train_pred_complex = model_complex.predict(X_train_complex, verbose=0)
lstm_valid_pred_complex = model_complex.predict(X_val_complex, verbose=0)

# Calculate and display RMSE
train_rmse_complex = np.sqrt(mean_squared_error(Y_train_complex, lstm_train_pred_complex))
valid_rmse_complex = np.sqrt(mean_squared_error(Y_val_complex, lstm_valid_pred_complex))
st.write(f"Train RMSE (Complex LSTM): {train_rmse_complex:.2f}")
st.write(f"Validation RMSE (Complex LSTM): {valid_rmse_complex:.2f}")

# Create valid DataFrame with predictions
valid_complex = pd.DataFrame(new_df[train_len:])
valid_complex['Predictions'] = lstm_valid_pred_complex

st.subheader('Complex LSTM Validation Predictions vs. Actuals')
fig_val_pred_complex, ax_val_pred_complex = plt.subplots(figsize=(16, 8))
ax_val_pred_complex.plot(valid_complex[['High', 'Predictions']])
ax_val_pred_complex.legend(['Validation Actuals', 'Predictions'])
ax_val_pred_complex.set_title('Complex LSTM Validation Predictions vs. Actuals')
ax_val_pred_complex.set_xlabel('Date')
ax_val_pred_complex.set_ylabel('High Price')
st.pyplot(fig_val_pred_complex)

# Create combined plot
train_complex_plot = new_df[:train_len]
st.subheader('Complex LSTM: Train, Validation, and Predictions Plot')
fig_combined_complex, ax_combined_complex = plt.subplots(figsize=(16, 8))
ax_combined_complex.plot(train_complex_plot, label='Train')
ax_combined_complex.plot(valid_complex[['High']], label='Validation Actuals')
ax_combined_complex.plot(valid_complex['Predictions'], label='Predictions')
ax_combined_complex.set_title('Complex LSTM Model: Train, Validation, and Predictions')
ax_combined_complex.set_xlabel('Date')
ax_combined_complex.set_ylabel('High Price')
ax_combined_complex.legend()
st.pyplot(fig_combined_complex)


st.subheader('Complex LSTM with Dropout Evaluation (10 runs)')
r1 = []
r2 = []
progress_text_dropout = "Running dropout LSTM models..."
my_bar_dropout = st.progress(0, text=progress_text_dropout)
for i in range(0, 10):
    my_bar_dropout.progress((i + 1) / 10, text=f"Running dropout LSTM model {i+1}/10")
    model_dropout = Sequential()
    model_dropout.add(LSTM(50, return_sequences=True, activation='relu', input_shape=(X_train_complex.shape[1], 1), recurrent_dropout=0.2))
    model_dropout.add(LSTM(50, return_sequences=False, activation='relu'))
    model_dropout.add(Dense(100))
    model_dropout.add(Dropout(0.2)) # Adding dropout layer
    model_dropout.add(Dense(25))
    model_dropout.add(Dense(1))
    opt1_dropout = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model_dropout.compile(loss='mean_squared_error', optimizer=opt1_dropout)
    model_dropout.fit(X_train_complex, Y_train_complex, epochs=100, batch_size=10, verbose=0)

    lstm_train_pred_dropout = model_dropout.predict(X_train_complex, verbose=0)
    lstm_valid_pred_dropout = model_dropout.predict(X_val_complex, verbose=0)
    r1.append(np.round(np.sqrt(mean_squared_error(Y_train_complex, lstm_train_pred_dropout)), 2))
    r2.append(np.round(np.sqrt(mean_squared_error(Y_val_complex, lstm_valid_pred_dropout)), 2))

st.success('Dropout LSTM evaluation complete!')

st.write("### Train RMSEs with Dropout (10 runs):")
st.write(r1)
st.write(f"Mean Train RMSE: {statistics.mean(r1):.2f}")
st.write(f"Standard Deviation Train RMSE: {statistics.stdev(r1):.2f}")

st.write("### Validation RMSEs with Dropout (10 runs):")
st.write(r2)
st.write(f"Mean Validation RMSE: {statistics.mean(r2):.2f}")
st.write(f"Standard Deviation Validation RMSE: {statistics.stdev(r2):.2f}")




st.subheader('SimpleRNN Model with Learning Rate Finder')
window = 10

# Prepare data (assuming X_train_complex, Y_train_complex, X_val_complex, Y_val_complex are already defined from previous steps)
model_rnn_lr_finder = Sequential()
model_rnn_lr_finder.add(SimpleRNN(50, return_sequences=True, activation='relu', input_shape=(X_train_complex.shape[1], 1)))
model_rnn_lr_finder.add(SimpleRNN(50, return_sequences=False, activation='relu'))
model_rnn_lr_finder.add(Dense(100))
model_rnn_lr_finder.add(Dense(25))
model_rnn_lr_finder.add(Dense(1))

lr_schedule = tensorflow.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-5 * 10**(epoch / 85))
opt_lr_finder = Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.7)
model_rnn_lr_finder.compile(loss='mean_squared_error', optimizer=opt_lr_finder)

st.text('Running Learning Rate Finder for SimpleRNN...')
with st.spinner('Training SimpleRNN for LR finder (epochs=100)...'):
    history = model_rnn_lr_finder.fit(X_train_complex, Y_train_complex, epochs=100, batch_size=10, verbose=0, callbacks=[lr_schedule])
st.success('Learning Rate Finder complete!')


st.subheader('Learning Rate vs. Loss for SimpleRNN')
fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
# Generate the learning rates manually for plotting
lrs = [1e-5 * 10**(e / 85) for e in range(len(history.history["loss"]))]
ax_lr.semilogx(lrs, history.history["loss"])
ax_lr.set_title('Learning Rate vs. Loss')
ax_lr.set_xlabel('Learning Rate')
ax_lr.set_ylabel('Loss')
ax_lr.axis([1e-5, 5e-4, 0, 1000]) # Set limits as in notebook
st.pyplot(fig_lr)


st.subheader('Optimized SimpleRNN Model Evaluation')

# Ensure window is set to 10 for consistency
window = 10

# Data preparation (assuming X_train_complex, Y_train_complex, X_val_complex, Y_val_complex are already defined)
# from previous steps, and train_len is consistent.

r1_optimized_rnn = []
r2_optimized_rnn = []

progress_text_optimized_rnn = "Running optimized SimpleRNN models..."
my_bar_optimized_rnn = st.progress(0, text=progress_text_optimized_rnn)

for i in range(0, 10):
    my_bar_optimized_rnn.progress((i + 1) / 10, text=f"Running optimized SimpleRNN model {i+1}/10")

    model_optimized_rnn = Sequential()
    model_optimized_rnn.add(SimpleRNN(50, return_sequences=True, activation='relu', input_shape=(X_train_complex.shape[1], 1)))
    model_optimized_rnn.add(SimpleRNN(50, return_sequences=False, activation='relu'))
    model_optimized_rnn.add(Dense(100))
    model_optimized_rnn.add(Dense(25))
    model_optimized_rnn.add(Dense(1))

    opt_optimized_rnn = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.7, clipvalue=1.0)
    model_optimized_rnn.compile(loss='mean_squared_error', optimizer=opt_optimized_rnn)

    model_optimized_rnn.fit(X_train_complex, Y_train_complex, epochs=100, batch_size=10, verbose=0)

    lstm_train_pred_optimized_rnn = model_optimized_rnn.predict(X_train_complex, verbose=0)
    lstm_valid_pred_optimized_rnn = model_optimized_rnn.predict(X_val_complex, verbose=0)

    r1_optimized_rnn.append(np.round(np.sqrt(mean_squared_error(Y_train_complex, lstm_train_pred_optimized_rnn)), 2))
    r2_optimized_rnn.append(np.round(np.sqrt(mean_squared_error(Y_val_complex, lstm_valid_pred_optimized_rnn)), 2))

st.success('Optimized SimpleRNN evaluation complete!')

st.write("### Train RMSEs (Optimized SimpleRNN - 10 runs):")
st.write(r1_optimized_rnn)
st.write(f"Mean Train RMSE: {statistics.mean(r1_optimized_rnn):.2f}")
st.write(f"Standard Deviation Train RMSE: {statistics.stdev(r1_optimized_rnn):.2f}")

st.write("### Validation RMSEs (Optimized SimpleRNN - 10 runs):")
st.write(r2_optimized_rnn)
st.write(f"Mean Validation RMSE: {statistics.mean(r2_optimized_rnn):.2f}")
st.write(f"Standard Deviation Train RMSE: {statistics.stdev(r2_optimized_rnn):.2f}")

# Create valid DataFrame with predictions from the last run
valid_optimized_rnn = pd.DataFrame(new_df[train_len:])
valid_optimized_rnn['Predictions'] = lstm_valid_pred_optimized_rnn

st.subheader('Optimized SimpleRNN Validation Predictions vs. Actuals')
fig_val_pred_optimized_rnn, ax_val_pred_optimized_rnn = plt.subplots(figsize=(16, 8))
ax_val_pred_optimized_rnn.plot(valid_optimized_rnn[['High', 'Predictions']])
ax_val_pred_optimized_rnn.legend(['Validation Actuals', 'Predictions'])
ax_val_pred_optimized_rnn.set_title('Optimized SimpleRNN Validation Predictions vs. Actuals')
ax_val_pred_optimized_rnn.set_xlabel('Date')
ax_val_pred_optimized_rnn.set_ylabel('High Price')
st.pyplot(fig_val_pred_optimized_rnn)



st.subheader('Future Price Prediction')

# Ensure window is set to 10 for consistency
window = 10

# Get the last 'window' days from new_df
last_window_days = new_df[-window:].values

# Reshape for model input
X_test = last_window_days.reshape(1, window, 1)

predicted_prices_future = []

# Predict the next 4 days iteratively
for _ in range(4):
    current_pred = model_optimized_rnn.predict(X_test, verbose=0)[0]
    predicted_prices_future.append(float(current_pred))
    # Update X_test for the next prediction: drop the oldest and add the new prediction
    X_test = np.append(X_test[:, 1:, :], current_pred.reshape(1, 1, 1), axis=1)

st.write(f"Predicted prices for the next 4 days: {predicted_prices_future}")

# Get actual prices for the last 4 days from the original df for comparison
actual_prices_comparison = df['High'].iloc[-4:].values

# Generate future dates for predicted prices
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=5, freq='D')[1:] # Next 4 days

# Create a DataFrame for comparison
future_comparison_data = {
    'Actual values': actual_prices_comparison,
    'Predicted values': predicted_prices_future
}
future_comparison_df = pd.DataFrame(future_comparison_data, index=future_dates)

st.subheader('Comparison of Actual and Predicted Future Prices')
st.write(future_comparison_df)

# Plotting the comparison
fig_future_pred, ax_future_pred = plt.subplots(figsize=(10, 6))
ax_future_pred.plot(future_comparison_df.index, future_comparison_df['Actual values'], label='Actual Prices', marker='o')
ax_future_pred.plot(future_comparison_df.index, future_comparison_df['Predicted values'], label='Predicted Prices', marker='x')
ax_future_pred.set_title('Future Price Prediction vs. Actuals')
ax_future_pred.set_xlabel('Date')
ax_future_pred.set_ylabel('High Price')
ax_future_pred.legend()
ax_future_pred.grid(True)
st.pyplot(fig_future_pred)

