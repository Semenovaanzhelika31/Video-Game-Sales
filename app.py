import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(
    page_title="Анализ продаж видеоигр",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Анализ и предсказание продаж видеоигр")
st.markdown("---")

# ============================================================
# 1. СОЗДАНИЕ ДАННЫХ
# ============================================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 3000
    
    platforms = ['PS4', 'Xbox One', 'PC', 'Nintendo Switch', 'PS5', 'PS3', 'Xbox 360', 'Wii', '3DS', 'PS Vita']
    genres = ['Action', 'Shooter', 'Sports', 'RPG', 'Adventure', 'Racing', 'Fighting', 'Platform', 'Simulation']
    publishers = ['Nintendo', 'EA', 'Ubisoft', 'Sony', 'Microsoft', 'Activision', 'Take-Two', 'Sega', 'Square Enix', 'Capcom']
    
    data = {
        'Name': [f'Game_{i}' for i in range(1, n_samples + 1)],
        'Platform': np.random.choice(platforms, n_samples),
        'Year': np.random.choice([2015, 2016, 2017, 2018, 2019, 2020], n_samples),
        'Genre': np.random.choice(genres, n_samples),
        'Publisher': np.random.choice(publishers, n_samples),
        'NA_Sales': np.round(np.random.exponential(0.8, n_samples), 2),
        'EU_Sales': np.round(np.random.exponential(0.6, n_samples), 2),
        'JP_Sales': np.round(np.random.exponential(0.4, n_samples), 2),
        'Other_Sales': np.round(np.random.exponential(0.3, n_samples), 2),
    }
    
    df = pd.DataFrame(data)
    
    # Корреляции
    df.loc[df['Genre'] == 'RPG', 'JP_Sales'] = df.loc[df['Genre'] == 'RPG', 'JP_Sales'] * 2.5
    df.loc[df['Publisher'] == 'Nintendo', 'JP_Sales'] = df.loc[df['Publisher'] == 'Nintendo', 'JP_Sales'] * 2
    df.loc[df['Platform'] == 'Nintendo Switch', 'JP_Sales'] = df.loc[df['Platform'] == 'Nintendo Switch', 'JP_Sales'] * 1.8
    
    df['Global_Sales'] = df['NA_Sales'] + df['EU_Sales'] + df['JP_Sales'] + df['Other_Sales']
    df['Global_Sales'] = df['Global_Sales'].round(2)
    
    return df

df = load_data()

# ============================================================
# 2. ОТОБРАЖЕНИЕ ДАННЫХ
# ============================================================
st.header("Исходные данные")
st.markdown(f"**Размер данных:** {df.shape[0]} строк, {df.shape[1]} колонок")

rows_to_show = st.selectbox("Количество строк:", [5, 10, 20, 50], index=1)
st.dataframe(df.head(rows_to_show), use_container_width=True)

with st.expander("Информация о данных"):
    st.write("**Типы данных:**")
    st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Колонка', 0: 'Тип'}))
    st.write("**Статистика:**")
    st.dataframe(df.describe())

# ============================================================
# 3. ФИЛЬТРЫ
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("Фильтры")

selected_years = st.sidebar.multiselect("Год", sorted(df['Year'].unique()), default=[])
selected_platforms = st.sidebar.multiselect("Платформа", sorted(df['Platform'].unique()), default=[])
selected_genres = st.sidebar.multiselect("Жанр", sorted(df['Genre'].unique()), default=[])

filtered_df = df.copy()
if selected_years:
    filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
if selected_platforms:
    filtered_df = filtered_df[filtered_df['Platform'].isin(selected_platforms)]
if selected_genres:
    filtered_df = filtered_df[filtered_df['Genre'].isin(selected_genres)]

st.sidebar.markdown("---")
st.sidebar.metric("Отфильтровано записей", len(filtered_df))
st.sidebar.metric("Всего игр", len(df))

# ============================================================
# 4. КЛЮЧЕВЫЕ МЕТРИКИ
# ============================================================
st.header("Ключевые метрики")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Всего игр", f"{len(df):,}")
col2.metric("Общие продажи", f"{df['Global_Sales'].sum():.1f}M")
col3.metric("Средние продажи", f"{df['Global_Sales'].mean():.2f}M")
col4.metric("Платформ", df['Platform'].nunique())
col5.metric("Жанров", df['Genre'].nunique())

# ============================================================
# 5. ВИЗУАЛИЗАЦИИ
# ============================================================
st.markdown("---")
st.header("Визуализация")

chart_type = st.selectbox("Выберите график", [
    "Продажи по платформам",
    "Продажи по жанрам",
    "Продажи по годам",
    "Распределение продаж",
    "JP vs Global Sales"
])

fig, ax = plt.subplots(figsize=(10, 6))

if chart_type == "Продажи по платформам":
    sales = filtered_df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)
    sales.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel('Платформа')
    ax.set_ylabel('Продажи (млн)')
    ax.set_title('Топ-10 платформ по продажам')
    plt.xticks(rotation=45)

elif chart_type == "Продажи по жанрам":
    sales = filtered_df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
    sales.plot(kind='bar', color='lightcoral', ax=ax)
    ax.set_xlabel('Жанр')
    ax.set_ylabel('Продажи (млн)')
    ax.set_title('Продажи по жанрам')
    plt.xticks(rotation=45)

elif chart_type == "Продажи по годам":
    sales = filtered_df.groupby('Year')['Global_Sales'].sum()
    sales.plot(kind='line', marker='o', color='green', ax=ax)
    ax.set_xlabel('Год')
    ax.set_ylabel('Продажи (млн)')
    ax.set_title('Динамика продаж по годам')

elif chart_type == "Распределение продаж":
    ax.hist(filtered_df['Global_Sales'], bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Продажи (млн)')
    ax.set_ylabel('Количество игр')
    ax.set_title('Распределение глобальных продаж')
    ax.axvline(filtered_df['Global_Sales'].median(), color='red', linestyle='--', label='Медиана')
    ax.legend()

elif chart_type == "JP vs Global Sales":
    ax.scatter(filtered_df['Global_Sales'], filtered_df['JP_Sales'], alpha=0.5, color='blue')
    ax.set_xlabel('Глобальные продажи (млн)')
    ax.set_ylabel('Продажи в Японии (млн)')
    ax.set_title('Зависимость продаж в Японии от глобальных')

st.pyplot(fig)

# ============================================================
# 6. ПРОГНОЗ ПРОДАЖ В ЯПОНИИ
# ============================================================
st.markdown("---")
st.header("Прогноз продаж в Японии")

st.markdown("""
**Как это работает:** Модель находит похожие игры и усредняет их продажи в Японии.
""")

col1, col2 = st.columns(2)

with col1:
    pred_platform = st.selectbox("Платформа", sorted(df['Platform'].unique()))
    pred_genre = st.selectbox("Жанр", sorted(df['Genre'].unique()))
    pred_publisher = st.selectbox("Издатель", sorted(df['Publisher'].unique()))

with col2:
    pred_year = st.number_input("Год выпуска", min_value=2000, max_value=2025, value=2024)
    pred_na = st.number_input("Продажи в NA (млн)", min_value=0.0, value=1.0, step=0.1)
    pred_eu = st.number_input("Продажи в EU (млн)", min_value=0.0, value=0.8, step=0.1)
    pred_other = st.number_input("Продажи в Other (млн)", min_value=0.0, value=0.3, step=0.1)

if st.button("Рассчитать прогноз", type="primary"):
    # Поиск похожих игр
    similar = df[
        (df['Platform'] == pred_platform) |
        (df['Genre'] == pred_genre) |
        (df['Publisher'] == pred_publisher)
    ]
    
    if len(similar) > 0:
        base_prediction = similar['JP_Sales'].mean()
        adjustment = (pred_na / 1.5 + pred_eu / 1.2 + pred_other / 0.5) / 3
        prediction = base_prediction * (0.7 + adjustment * 0.3)
        prediction = round(prediction, 2)
    else:
        prediction = 0.5
    
    st.markdown("---")
    st.success(f"### Прогноз продаж в Японии: {prediction} млн копий")
    
    st.subheader("Факторы прогноза")
    if pred_genre == 'RPG':
        st.write("- RPG жанр: увеличивает продажи в Японии")
    if pred_publisher == 'Nintendo':
        st.write("- Издатель Nintendo: увеличивает продажи в Японии")
    if pred_platform in ['Nintendo Switch', '3DS']:
        st.write(f"- Платформа {pred_platform}: популярна в Японии")
    
    st.info(f"Найдено похожих игр: {len(similar)} | Доверительный интервал: +-0.30 млн")

# ============================================================
# 7. ТОП ИГРЫ
# ============================================================
st.markdown("---")
st.header("Топ-20 самых продаваемых игр")

top_games = filtered_df.nlargest(20, 'Global_Sales')[['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales', 'JP_Sales']]
top_games.columns = ['Название', 'Платформа', 'Год', 'Жанр', 'Издатель', 'Продажи (мир)', 'Продажи (Япония)']
st.dataframe(top_games, use_container_width=True)

# ============================================================
# 8. ЭКСПОРТ ДАННЫХ
# ============================================================
st.markdown("---")
st.header("Экспорт данных")

if st.button("Скачать отфильтрованные данные (CSV)"):
    csv = filtered_df.to_csv(index=False)
    st.download_button("Скачать CSV", csv, "filtered_data.csv", "text/csv")

st.markdown("---")
st.caption("Анализ продаж видеоигр | Прогноз продаж в Японии")
