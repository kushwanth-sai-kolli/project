import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from scipy.stats import yeojohnson
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📊 Hotel Booking Time Series Forecasting App")

uploaded_file = st.file_uploader("Upload hotel_bookings.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['children'] = df['children'].fillna(df['children'].median())
    df['country'] = df['country'].fillna(df['country'].mode()[0])
    df['agent'] = df['agent'].notna().astype(int)
    df['company'] = df['company'].notna().astype(int)
    df.dropna(inplace=True)

    df['hotel'] = df['hotel'].replace({'Resort Hotel': 0, 'City Hotel': 1}).astype(int)
    df['arrival_date_month'] = df['arrival_date_month'].replace({
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }).astype(int)

    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'].astype(str) + '-' +
        df['arrival_date_day_of_month'].astype(str),
        errors='coerce'
    )

    df['stay_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['booking_price'] = df['adr'] * df['stay_duration']
    df.drop(columns=['adr'], inplace=True)

    # Apply Yeo-Johnson transformation
    exclude_cols = ['arrival_date_year', 'arrival_date_month']
    df_transformed = df.copy()
    numeric_cols = df_transformed.select_dtypes(include=['number']).columns
    transform_cols = [col for col in numeric_cols if col not in exclude_cols]

    for col in transform_cols:
        df_transformed[col], _ = yeojohnson(df_transformed[col])

    # Restore columns
    df_transformed['arrival_date_year'] = df['arrival_date_year']
    df_transformed['arrival_date_month'] = df['arrival_date_month']
    df_transformed['arrival_date'] = df['arrival_date']

    # Prepare monthly transformed time series
    df_transformed = df_transformed[['arrival_date', 'booking_price']].dropna()
    df_transformed = df_transformed.rename(columns={'arrival_date': 'ds', 'booking_price': 'y'})
    df_transformed = df_transformed.sort_values('ds')
    monthly = df_transformed.set_index('ds').resample('M').sum()['y'].dropna()

    st.subheader("1️⃣ Preview Cleaned & Resampled Data")
    st.line_chart(monthly)

    st.subheader("2️⃣ Time Series Decomposition")
    model_type = st.radio("Choose Decomposition Type", ["Additive", "Multiplicative"])
    freq = st.slider("Select Seasonal Period (e.g., 30 for monthly)", 1, 365, 30)

    try:
        decomposition = seasonal_decompose(monthly, model=model_type.lower(), period=freq)
        fig, ax = plt.subplots(4, 1, figsize=(10, 8))
        decomposition.observed.plot(ax=ax[0], title="Observed")
        decomposition.trend.plot(ax=ax[1], title="Trend")
        decomposition.seasonal.plot(ax=ax[2], title="Seasonality")
        decomposition.resid.plot(ax=ax[3], title="Residuals")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not perform decomposition: {e}")

    st.subheader("3️⃣ Forecasting & Accuracy Evaluation")
    model_choice = st.selectbox("Choose Forecasting Model", ["Holt's Linear Trend", "ARIMA", "Prophet", "LSTM"])

    train = monthly.iloc[:-6]
    test = monthly.iloc[-6:]
    forecast, actual = None, None

    if model_choice == "Holt's Linear Trend":
        model = Holt(train).fit()
        forecast = model.forecast(steps=len(test))
        actual = test.values

    elif model_choice == "ARIMA":
        model = ARIMA(train, order=(1,1,1)).fit()
        forecast = model.forecast(steps=len(test))
        actual = test.values

    elif model_choice == "Prophet":
        prophet_df = pd.DataFrame({'ds': monthly.index, 'y': monthly.values})
        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=6, freq='M')
        forecast_df = m.predict(future)
        forecast = forecast_df[-6:]['yhat'].values
        actual = monthly[-6:].values

        fig3 = m.plot(forecast_df)
        st.pyplot(fig3)

    elif model_choice == "LSTM":
        look_back = 12
        data_lstm = monthly.values.reshape(-1, 1)

        scaler_lstm = MinMaxScaler()
        scaled_data = scaler_lstm.fit_transform(data_lstm)

        generator = TimeseriesGenerator(scaled_data, scaled_data, length=look_back, batch_size=1)

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(generator, epochs=10, verbose=0)

        predictions = []
        current_batch = scaled_data[-look_back:].reshape(1, look_back, 1)
        for i in range(6):
            pred = model.predict(current_batch)[0]
            predictions.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

        forecast = scaler_lstm.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        actual = monthly[-6:].values

    if forecast is not None and actual is not None:
        st.write("🔮 Forecast vs Actual")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(monthly.index[-12:], monthly[-12:], label="Actual")
        ax2.plot(monthly.index[-6:], forecast, label="Forecast")
        ax2.set_title(f"{model_choice} Forecast (Next 6 Months)")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("📏 Evaluation Metrics")
        mae = mean_absolute_error(actual, forecast)
        mse = mean_squared_error(actual, forecast)
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
