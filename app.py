import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------------
# 1. Загрузка и предобработка данных (кэшируем для скорости)
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("vgsales.csv")
    df = df.drop(columns=["Rank"], errors="ignore")
    df = df.dropna(subset=["Year", "Global_Sales"])
    df["Year"] = df["Year"].astype(int)
    # Оставляем нужные колонки
    df = df[["Name", "Platform", "Year", "Genre", "Publisher",
             "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]]
    return df

df = load_data()

# -------------------------------------------------------------------
# 2. Боковая панель с фильтрами (глобальные для всего приложения)
# -------------------------------------------------------------------
st.sidebar.title(" Фильтры")

# Мультивыбор платформ
platforms = sorted(df["Platform"].unique())
selected_platforms = st.sidebar.multiselect("Платформа", platforms, default=platforms[:5])

# Мультивыбор жанров
genres = sorted(df["Genre"].unique())
selected_genres = st.sidebar.multiselect("Жанр", genres, default=genres)

# Диапазон лет
min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider("Диапазон лет", min_year, max_year, (2000, 2020))

# Применяем фильтры
filtered_df = df[
    (df["Platform"].isin(selected_platforms)) &
    (df["Genre"].isin(selected_genres)) &
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1])
]

# -------------------------------------------------------------------
# 3. Заголовок и выбор дашборда
# -------------------------------------------------------------------
st.title(" Анализ продаж видеоигр")
dashboard = st.radio("Выберите дашборд:", [" Общий обзор рынка", " Анализ платформ и жанров"], horizontal=True)

# -------------------------------------------------------------------
# 4. Дашборд 1: Общий обзор рынка
# -------------------------------------------------------------------
if dashboard == " Общий обзор рынка":
    st.header(" Общий обзор рынка")

    # 4.1 Динамика продаж по годам
    st.subheader("Динамика глобальных продаж по годам")
    yearly = filtered_df.groupby("Year")["Global_Sales"].sum().reset_index()
    fig1 = px.line(yearly, x="Year", y="Global_Sales", markers=True,
                   title="Суммарные продажи по годам",
                   labels={"Global_Sales": "Продажи (млн копий)"})
    st.plotly_chart(fig1, use_container_width=True)

    # 4.2 Региональные продажи (столбцы)
    st.subheader("Продажи по регионам")
    region_sales = {
        "Регион": ["NA", "EU", "JP", "Other"],
        "Продажи": [
            filtered_df["NA_Sales"].sum(),
            filtered_df["EU_Sales"].sum(),
            filtered_df["JP_Sales"].sum(),
            filtered_df["Other_Sales"].sum()
        ]
    }
    df_reg = pd.DataFrame(region_sales)
    fig2 = px.bar(df_reg, x="Регион", y="Продажи", text="Продажи",
                  title="Суммарные продажи по регионам",
                  labels={"Продажи": "Продажи (млн копий)"})
    st.plotly_chart(fig2, use_container_width=True)

    # 4.3 Топ-5 издателей за последние 10 лет
    st.subheader("Топ-5 издателей (последние 10 лет)")
    last10 = filtered_df[filtered_df["Year"] >= 2016]
    top_publishers = last10.groupby("Publisher")["Global_Sales"].sum().nlargest(5).reset_index()
    fig3 = px.bar(top_publishers, x="Global_Sales", y="Publisher", orientation='h',
                  title="Топ-5 издателей (2016–2020)",
                  labels={"Global_Sales": "Продажи (млн копий)"})
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------------------------
# 5. Дашборд 2: Анализ платформ и жанров
# -------------------------------------------------------------------
elif dashboard == " Анализ платформ и жанров":
    st.header(" Анализ платформ и жанров")

    # 5.1 Жизненный цикл платформ (только для выбранных платформ)
    st.subheader("Жизненный цикл платформ (годы после запуска)")
    # Отбираем только платформы, выбранные пользователем (до 6 для читаемости)
    plot_platforms = [p for p in selected_platforms if p in df["Platform"].unique()]
    if len(plot_platforms) > 6:
        st.warning("Для наглядности отображены первые 6 платформ")
        plot_platforms = plot_platforms[:6]

    # Рассчитываем годы после запуска для каждой платформы
    lifecycle_data = []
    for plat in plot_platforms:
        plat_df = filtered_df[filtered_df["Platform"] == plat]
        if plat_df.empty:
            continue
        first_year = plat_df["Year"].min()
        # Группируем по году и суммируем продажи
        yearly_plat = plat_df.groupby("Year")["Global_Sales"].sum().reset_index()
        yearly_plat["YearsSinceLaunch"] = yearly_plat["Year"] - first_year
        yearly_plat["Platform"] = plat
        lifecycle_data.append(yearly_plat)
    if lifecycle_data:
        df_life = pd.concat(lifecycle_data, ignore_index=True)
        # Нормализуем к пику
        peak = df_life.groupby("Platform")["Global_Sales"].transform('max')
        df_life["PctOfPeak"] = df_life["Global_Sales"] / peak * 100
        fig4 = px.line(df_life, x="YearsSinceLaunch", y="PctOfPeak", color="Platform",
                       title="Процент от пика продаж по годам после запуска",
                       labels={"PctOfPeak": "% от пика", "YearsSinceLaunch": "Годы после запуска"})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Нет данных для выбранных платформ")

    # 5.2 Сравнение Action vs Shooter по регионам
    st.subheader("Action vs Shooter по регионам")
    # Фильтруем только эти жанры
    genre_df = filtered_df[filtered_df["Genre"].isin(["Action", "Shooter"])]
    if not genre_df.empty:
        # Агрегация по регионам и жанрам
        action_shooter = []
        for genre in ["Action", "Shooter"]:
            gdf = genre_df[genre_df["Genre"] == genre]
            action_shooter.append({
                "Genre": genre,
                "NA": gdf["NA_Sales"].sum(),
                "EU": gdf["EU_Sales"].sum(),
                "JP": gdf["JP_Sales"].sum()
            })
        df_as = pd.DataFrame(action_shooter)
        # Переводим в длинный формат для plotly
        df_as_long = df_as.melt(id_vars="Genre", var_name="Region", value_name="Sales")
        fig5 = px.bar(df_as_long, x="Region", y="Sales", color="Genre", barmode="group",
                      title="Сравнение Action и Shooter по регионам",
                      labels={"Sales": "Продажи (млн копий)"})
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Нет данных для жанров Action/Shooter при текущих фильтрах")

    # 5.3 (Опционально) Тепловая карта Жанр × Платформа
    st.subheader("Продажи по жанрам и платформам (топ-6 жанров, топ-10 платформ)")
    top_genres = filtered_df.groupby("Genre")["Global_Sales"].sum().nlargest(6).index
    top_platforms = filtered_df.groupby("Platform")["Global_Sales"].sum().nlargest(10).index
    heat_df = filtered_df[filtered_df["Genre"].isin(top_genres) & filtered_df["Platform"].isin(top_platforms)]
    pivot_heat = heat_df.pivot_table(index="Genre", columns="Platform", values="Global_Sales", aggfunc="sum", fill_value=0)
    fig6 = px.imshow(pivot_heat, text_auto=True, aspect="auto",
                     title="Тепловая карта продаж (млн копий)",
                     color_continuous_scale="Viridis")
    st.plotly_chart(fig6, use_container_width=True)

# -------------------------------------------------------------------
# 6. Дополнительная информация
# -------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info(f" Текущий диапазон: {len(filtered_df)} игр, сумма продаж: {filtered_df['Global_Sales'].sum():.1f} млн копий")
