#Importemos nuestras herramientas para esta página

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("📈 Forecast de Ventas con LSTM")

st.markdown("Modelo LSTM entrenado en tiempo real.")

# Cargar datos
df = pd.read_csv("C:\\Users\\USUARIO\\Pryoectos ML\\Red Neuronal Ventas\\Data\\train1.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Ingreso de Datos
st.sidebar.header("Seleccionar Parámetros")
#Selelciona solamente una tienda y depto
store = st.sidebar.selectbox("Store", sorted(df["Store"].unique()))
dept = st.sidebar.selectbox("Department", sorted(df["Dept"].unique()))

#Filtrado de datos
df_filtrado = df[(df["Store"] == store) & (df["Dept"] == dept)]
df_filtrado = df_filtrado.sort_values("Date")

#Condicional encargada de generar el forecast y de crear la secuencia
if st.sidebar.button("Generar Forecast"):

    if len(df_filtrado) < 60:
        st.warning("No hay suficientes datos para entrenar la LSTM.")
        st.stop()

    # Solo ventas
    data = df_filtrado["Weekly_Sales"].values.reshape(-1,1)

    # Escalado
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Crear secuencias
    def crear_secuencias(data, pasos=12):
        X, y = [], []
        for i in range(len(data) - pasos):
            X.append(data[i:i+pasos])
            y.append(data[i+pasos])
        return np.array(X), np.array(y)

    pasos = 12
    X, y = crear_secuencias(data_scaled, pasos)

    # División train/test
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Modelo LSTM simple
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(pasos,1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    with st.spinner("Entrenando modelo LSTM..."):
        model.fit(X_train, y_train,
                  epochs=20,
                  batch_size=16,
                  verbose=0)

    st.success("Modelo entrenado correctamente.")

    # Forecast futuro
    n_forecast = 12
    ultima_secuencia = data_scaled[-pasos:]
    current_batch = ultima_secuencia.reshape((1, pasos, 1))

    forecast = []

    for i in range(n_forecast):
        pred = model.predict(current_batch, verbose=0)[0]
        forecast.append(pred)
        current_batch = np.append(current_batch[:,1:,:],
                                  [[pred]],
                                  axis=1)

    forecast = scaler.inverse_transform(forecast)

    # Fechas futuras
    ultima_fecha = df_filtrado["Date"].max()
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.Timedelta(weeks=1),
        periods=n_forecast,
        freq="W-FRI"
    )

    # Gráfico
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(df_filtrado["Date"],
            df_filtrado["Weekly_Sales"],
            label="Histórico")

    ax.plot(fechas_futuras,
            forecast,
            linestyle="--",
            label="Forecast LSTM")

    ax.legend()
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Ventas")

    st.pyplot(fig)

    # Tabla
    forecast_df = pd.DataFrame({
        "Fecha": fechas_futuras,
        "Ventas_Pronosticadas": forecast.flatten()
    })

    st.subheader("📋 Predicciones")
    st.dataframe(forecast_df)
