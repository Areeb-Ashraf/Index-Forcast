# Imports
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Get data
START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Index Forecasting')

# Define a dictionary for mapping names to symbols
index_mapping = {'Nasdaq Index': '^NDX',
                 'Nifty Index': '^NSEI',
                 'Nasdaq Technology Index': '^NDXT',
                 'Nifty Technology Index': '^CNXIT'}

# Select index name
selected_name = st.selectbox('Select dataset for prediction', list(index_mapping.keys()))

# Get the corresponding symbol
selected_index = index_mapping[selected_name]

# Get months of prediction
n_months = st.slider('Months of prediction:', 1, 12)
period = n_months * 30  # Assuming an average of 30 days in a month

# Cache data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text('Loading data...')
data = load_data(selected_index)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Evaluate the model
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:]
test['forecast'] = forecast['yhat'][-len(test):].values
mae = (test['forecast'] - test['Close']).abs().mean()

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_months} month(s)')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Display evaluation metrics
st.subheader('Model Accuracy Metrics')
st.write(f'Mean Absolute Error (MAE): {mae:.2f}')