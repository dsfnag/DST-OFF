import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from pandas.tseries.offsets import MonthEnd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn import metrics
import pandas_profiling

import streamlit as st
import plotly.express as px

st.title("Модель 'Анализ транзакций'\nВерсия 20.11.14.1")

st.header("Загрузка и предобработка данных")
df_agg = pd.read_csv('aggregates.csv')
df_trx = pd.read_csv('transactions.csv')

# profile = pandas_profiling.ProfileReport(df_trx)
# profile.to_file("core/templates/output.html")

# Добавляем колонку с последним днем месяца
df_agg['last_day_month'] = pd.to_datetime(df_agg['report_date']) + MonthEnd(1)

# Переводим в дату
df_trx['last_day_month'] = pd.to_datetime(df_trx['last_day_month'])

# Объединяем в один датасет
df = df_trx.merge(df_agg, how='inner',left_on=["client_id","last_day_month"], right_on=["client_id","last_day_month"])

st.markdown("Датасет. Первые 100 записей")
st.write(df.head(100))

st.header("Считаем количество транзакций за текущий и прошлый месяц и их отклонение")

# Группируем данные по клиентам и месяцам
df_month_agg = df.groupby(by=["client_id","last_day_month"], as_index=False)["att_cnt"].sum()

# Выводим колонку с предыдущим месяцем
df_month_agg["prev_month"] = pd.to_datetime(df_month_agg["last_day_month"].values.astype('datetime64[M]')) - timedelta(days=1)

# Соединяем в один датасет данные за текущий и прошлый месяц
df_month_agg = df_month_agg.merge(df_month_agg, how = 'left', left_on = ["prev_month","client_id"], right_on = ["last_day_month","client_id"])

# Выводим целевые показатели
# Процентное отклонение
df_month_agg['att_cnt_change %'] = ((df_month_agg["att_cnt_x"]-df_month_agg["att_cnt_y"])/df_month_agg["att_cnt_y"])

# Отклик
df_month_agg['att_cnt_grouth'] = np.where(df_month_agg['att_cnt_change %']>= 0.3, True, False).astype('float64')

# Отток
df_month_agg['att_cnt_sink'] = np.where(df_month_agg['att_cnt_change %']<= -0.3, True, False).astype('float64')

# Оставляем только нужные колонки
df_month_agg = df_month_agg[["client_id","last_day_month_x",'att_cnt_change %','att_cnt_sink',"att_cnt_grouth"]]

# Соединяем в один датасет
df = df.merge(df_month_agg,how = 'left', left_on = ["client_id","last_day_month"], right_on = ["client_id","last_day_month_x"])

st.write(df.head(100))

def df_types(df_input):
    df_output = pd.DataFrame(
        df_input[df_input[c].notna()][c].apply(type).value_counts() for c in df_input.columns
    ).fillna(0).astype(int)
    df_output['NaN'] = df_input.isna().sum()
    df_output['sum'] = df_output.sum(axis=1)
    return df_output

st.header("Выводим типы данных и пропуски")
st.write(df_types(df).head(60))













@st.cache
def get_data():
    return pd.read_csv("http://data.insideairbnb.com/united-states/ny/new-york-city/2019-09-12/visualisations/listings.csv")

df = get_data()

st.markdown("Welcome to this in-depth introduction to [Streamlit](www.streamlit.io)! For this exercise, we'll use an Airbnb [dataset](http://data.insideairbnb.com/united-states/ny/new-york-city/2019-09-12/visualisations/listings.csv) containing NYC listings.")
st.header("Customary quote")
st.markdown("> I just love to go home, no matter where I am, the most luxurious hotel suite in the world, I love to go home.\n\nMichael Caine")
st.header("Airbnb NYC listings: data at a glance")
st.markdown("The first five records of the Airbnb data we downloaded.")
st.dataframe(df.head())
st.header("Caching our data")
st.markdown("Streamlit has a handy decorator [`st.cache`](https://streamlit.io/docs/api.html#optimize-performance) to enable data caching.")
st.code("""
@st.cache
def get_data():
    url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-09-12/visualisations/listings.csv"
    return pd.read_csv(url)
""", language="python")
st.markdown("_To display a code block, pass in the string to display as code to [`st.code`](https://streamlit.io/docs/api.html#streamlit.code)_.")
with st.echo():
    st.markdown("Alternatively, use [`st.echo`](https://streamlit.io/docs/api.html#streamlit.echo).")

st.header("Where are the most expensive properties located?")
st.subheader("On a map")
st.markdown("The following map shows the top 1% most expensive Airbnbs priced at $800 and above.")
st.map(df.query("price>=800")[["latitude", "longitude"]].dropna(how="any"))
st.subheader("In a table")
st.markdown("Following are the top five most expensive properties.")
st.write(df.query("price>=800").sort_values("price", ascending=False).head())

st.subheader("Selecting a subset of columns")
st.write(f"Out of the {df.shape[1]} columns, you might want to view only a subset. Streamlit has a [multiselect](https://streamlit.io/docs/api.html#streamlit.multiselect) widget for this.")
defaultcols = ["name", "host_name", "neighbourhood", "room_type", "price"]
cols = st.multiselect("Columns", df.columns.tolist(), default=defaultcols)
st.dataframe(df[cols].head(10))

st.header("Average price by room type")
st.write("You can also display static tables. As opposed to a data frame, with a static table you cannot sorting by clicking a column header.")
st.table(df.groupby("room_type").price.mean().reset_index()\
    .round(2).sort_values("price", ascending=False)\
    .assign(avg_price=lambda x: x.pop("price").apply(lambda y: "%.2f" % y)))

st.header("Which host has the most properties listed?")
listingcounts = df.host_id.value_counts()
top_host_1 = df.query('host_id==@listingcounts.index[0]')
top_host_2 = df.query('host_id==@listingcounts.index[1]')
st.write(f"""**{top_host_1.iloc[0].host_name}** is at the top with {listingcounts.iloc[0]} property listings.
**{top_host_2.iloc[1].host_name}** is second with {listingcounts.iloc[1]} listings. Following are randomly chosen
listings from the two displayed as JSON using [`st.json`](https://streamlit.io/docs/api.html#streamlit.json).""")

st.json({top_host_1.iloc[0].host_name: top_host_1\
    [["name", "neighbourhood", "room_type", "minimum_nights", "price"]]\
        .sample(2, random_state=4).to_dict(orient="records"),
        top_host_2.iloc[0].host_name: top_host_2\
    [["name", "neighbourhood", "room_type", "minimum_nights", "price"]]\
        .sample(2, random_state=4).to_dict(orient="records")})

st.header("What is the distribution of property price?")
st.write("""Select a custom price range from the side bar to update the histogram below displayed as a Plotly chart using
[`st.plotly_chart`](https://streamlit.io/docs/api.html#streamlit.plotly_chart).""")
values = st.sidebar.slider("Price range", float(df.price.min()), float(df.price.clip(upper=1000.).max()), (50., 300.))
f = px.histogram(df.query(f"price.between{values}"), x="price", nbins=15, title="Price distribution")
f.update_xaxes(title="Price")
f.update_yaxes(title="No. of listings")
st.plotly_chart(f)

st.header("What is the distribution of availability in various neighborhoods?")
st.write("Using a radio button restricts selection to only one option at a time.")
st.write("Notice how we use a static table below instead of a data frame. \
Unlike a data frame, if content overflows out of the section margin, \
a static table does not automatically hide it inside a scrollable area. \
Instead, the overflowing content remains visible.")
neighborhood = st.radio("Neighborhood", df.neighbourhood_group.unique())
show_exp = st.checkbox("Include expensive listings")
show_exp = " and price<200" if not show_exp else ""

@st.cache
def get_availability(show_exp, neighborhood):
    return df.query(f"""neighbourhood_group==@neighborhood{show_exp}\
        and availability_365>0""").availability_365.describe(\
            percentiles=[.1, .25, .5, .75, .9, .99]).to_frame().T

st.table(get_availability(show_exp, neighborhood))
st.write("At 169 days, Brooklyn has the lowest average availability. At 226, Staten Island has the highest average availability.\
    If we include expensive listings (price>=$200), the numbers are 171 and 230 respectively.")
st.markdown("_**Note:** There are 18431 records with `availability_365` 0 (zero), which I've ignored._")

df.query("availability_365>0").groupby("neighbourhood_group")\
    .availability_365.mean().plot.bar(rot=0).set(title="Average availability by neighborhood group",
        xlabel="Neighborhood group", ylabel="Avg. availability (in no. of days)")
