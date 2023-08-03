import time
import matplotlib
import streamlit as st
from datetime import date
import yfinance as yf
from neuralprophet import NeuralProphet
matplotlib.use('Agg')
from plotly import graph_objects as go
import pandas as pd
from stockstats import StockDataFrame as Sdf

#hide_menu_style = """
       # <style>
        #MainMenu {visibility: hidden;}
       # </style>
       # """
#st.markdown(hide_menu_style, unsafe_allow_html=True)

START = "2022-05-01"
# TODAY = date.today().strftime("%Y-%m-%d")
TODAY = date.fromtimestamp(time.time())
# st.title("Stock Prediction App")
AllStocks = pd.read_excel("tumhisse.xlsx", sheet_name=0)
# print((stocks['Hisse']))
stocks = AllStocks['Hisse']
print(TODAY)

# stocks = ("FROTO.IS", "GARAN.IS", "TCELL.IS", "CCOLA.IS","KUTPO.IS","TOASO.IS","TUPRS.IS","ASELS.IS","ALARK.IS","TMSN.IS","TTRAK.IS","LOGO.IS","SAHOL.IS","DEVA.IS","MGROS.IS","KOZAL.IS", "KORDS.IS","THYAO.IS","BRYAT.IS","SISE.IS","PETKM.IS","EKGYO.IS","ARCLK.IS","TAVHL.IS","SASA.IS","KRDMD.IS","VESTL.IS","BFREN.IS","HEKTS.IS","BIMAS.IS","OYAKC.IS","MAVI.IS","JANTS.IS","GWIND.IS","DOHOL.IS","AEFES.IS","ISCTR.IS","ZOREN.IS","TRILC.IS","TSKB.IS","TURSG.IS","AGHOL.IS","AKBNK.IS")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)
n_years = st.slider("Years of prediction", 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, period=None)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done")


def sdf_data(ticker):
    data = yf.download(ticker, START, period=None, interval="1d")
    stock_df = Sdf.retype(data)
    return stock_df['rsi_14']


#for stock in stocks:
   # print (stock, sdf_data(stock).tail())

stock_df = sdf_data(selected_stocks)

st.write(stock_df.tail())

st.subheader('Raw data')
st.write(data.tail())
st.write(yf.Ticker(selected_stocks).actions)


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
print(df_train)
df_train['ds'] = df_train['ds'].dt.tz_localize(None)
print(df_train)
m = NeuralProphet(epochs=100,
                  yearly_seasonality=True,
                  weekly_seasonality=False,
                  daily_seasonality=False,
                  batch_size=64,
                  learning_rate=1.0,
                  )

metrics = m.fit(df_train, freq="D")
future = m.make_future_dataframe(df_train, periods=period, n_historic_predictions=len(df_train))

forecast = m.predict(future)
# forecast.head()

print(forecast)
# forecast = forecast.rename({'y':'Fiyat','yhat1':'Tahmin'},axis=1)

st.subheader('Forecast data')
st.write(forecast.tail())
st.write('Forecast data')
# forecast = forecast.rename({'Fiyat':'y','Tahmin':'yhat1'},axis=1)
# fig1 = m.plot(forecast,ylabel='Fiyat',xlabel='Tarih', )
fig1 = m.plot(forecast)
# forecast = forecast.rename({'y':'Fiyat','yhat1':'Tahmin'},axis=1)

st.plotly_chart(fig1)
# plt.savefig('fig1.png')
# image = Image.open('fig1.png')
# st.image(image)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
