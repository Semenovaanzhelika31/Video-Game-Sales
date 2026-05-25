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
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from io import BytesIO

st.set_page_config(
    page_title="Анализ продаж видеоигр",
    page_icon=":video_game:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Анализ и предсказание продаж видеоигр")
st.markdown("---")

# ============================================================
# 1. СОЗДАНИЕ ДАННЫХ С РЕАЛИСТИЧНЫМИ ЗАВИСИМОСТЯМИ
# ============================================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 15000
    
    platforms = ['PS4', 'Xbox One', 'PC', 'Nintendo Switch', 'PS5', 'PS3', 'Xbox 360', 'Wii', '3DS', 'PS Vita']
    genres = ['Action', 'Shooter', 'Sports', 'RPG', 'Adventure', 'Racing', 'Fighting', 'Platform', 'Simulation']
    publishers = ['Nintendo', 'EA', 'Ubisoft', 'Sony', 'Microsoft', 'Activision', 'Take-Two', 'Sega', 'Square Enix', 'Capcom']
    
    # Базовые продажи
    data = {
        'Name': [f'Game_{i}' for i in range(1, n_samples + 1)],
        'Platform': np.random.choice(platforms, n_samples),
        'Year': np.random.choice([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020], n_samples),
        'Genre': np.random.choice(genres, n_samples),
        'Publisher': np.random.choice(publishers, n_samples),
    }
    df = pd.DataFrame(data)
    
    # Базовые продажи с шумом
    base_na = np.random.exponential(0.5, n_samples)
    base_eu = np.random.exponential(0.4, n_samples)
    base_jp = np.random.exponential(0.3, n_samples)
    base_other = np.random.exponential(0.2, n_samples)
    
    # Коэффициенты влияния (более сильные)
    jp_multiplier = pd.Series(1.0, index=range(n_samples))
    na_multiplier = pd.Series(1.0, index=range(n_samples))
    eu_multiplier = pd.Series(1.0, index=range(n_samples))
    
    # Влияние жанра на JP_Sales
    genre_jp = {
        'RPG': 4.5,
        'Platform': 3.0,
        'Action': 1.8,
        'Adventure': 1.5,
        'Fighting': 2.0,
        'Racing': 1.2,
        'Sports': 0.8,
        'Shooter': 0.5,
        'Simulation': 1.3
    }
    
    # Влияние жанра на NA_Sales
    genre_na = {
        'Shooter': 3.5,
        'Sports': 2.5,
        'Action': 2.0,
        'Racing': 1.5,
        'Adventure': 1.2,
        'RPG': 1.0,
        'Fighting': 1.0,
        'Platform': 0.8,
        'Simulation': 0.7
    }
    
    # Влияние издателя на JP_Sales
    publisher_jp = {
        'Nintendo': 4.0,
        'Square Enix': 2.5,
        'Capcom': 2.2,
        'Sega': 1.8,
        'Sony': 1.5,
        'Bandai Namco': 2.0,
        'Konami': 1.6,
        'EA': 0.6,
        'Ubisoft': 0.7,
        'Microsoft': 0.5,
        'Activision': 0.4,
        'Take-Two': 0.5
    }
    
    # Влияние платформы на JP_Sales
    platform_jp = {
        'Nintendo Switch': 3.5,
        '3DS': 3.0,
        'PS4': 1.5,
        'PS5': 1.4,
        'PS3': 1.2,
        'Wii': 2.5,
        'PS Vita': 1.8,
        'PC': 0.6,
        'Xbox One': 0.3,
        'Xbox 360': 0.3
    }
    
    # Применяем коэффициенты
    for i in range(n_samples):
        genre = df.loc[i, 'Genre']
        publisher = df.loc[i, 'Publisher']
        platform = df.loc[i, 'Platform']
        
        jp_multiplier[i] *= genre_jp.get(genre, 1.0)
        jp_multiplier[i] *= publisher_jp.get(publisher, 1.0)
        jp_multiplier[i] *= platform_jp.get(platform, 1.0)
        
        na_multiplier[i] *= genre_na.get(genre, 1.0)
        eu_multiplier[i] *= genre_na.get(genre, 0.8)
    
    # Добавляем случайный шум
    jp_multiplier = jp_multiplier * np.random.normal(1.0, 0.2, n_samples)
    na_multiplier = na_multiplier * np.random.normal(1.0, 0.2, n_samples)
    eu_multiplier = eu_multiplier * np.random.normal(1.0, 0.2, n_samples)
    
    df['NA_Sales'] = (base_na * na_multiplier).round(2)
    df['EU_Sales'] = (base_eu * eu_multiplier).round(2)
    df['JP_Sales'] = (base_jp * jp_multiplier).round(2)
    df['Other_Sales'] = (base_other * np.random.normal(1.0, 0.3, n_samples)).round(2)
    df['Global_Sales'] = (df['NA_Sales'] + df['EU_Sales'] + df['JP_Sales'] + df['Other_Sales']).round(2)
    
    # Ограничиваем максимальные значения
    for col in ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']:
        df[col] = df[col].clip(upper=50)
    
    return df

df = load_data()

# ============================================================
# 2. ОТОБРАЖЕНИЕ ДАННЫХ
# ============================================================
st.header("Исходные данные")
st.markdown(f"**Размер данных:** {df.shape[0]} строк, {df.shape[1]} колонок")

rows_to_show = st.selectbox("Количество строк для отображения:", [5, 10, 20, 50, 100], index=1)
st.dataframe(df.head(rows_to_show), use_container_width=True)

with st.expander("Информация о данных"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Типы данных:**")
        st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Колонка', 0: 'Тип'}))
    with col2:
        st.write("**Основная статистика:**")
        st.dataframe(df.describe())

# ============================================================
# 3. ФИЛЬТРЫ
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("Фильтры")

years = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect("Год выпуска", years, default=years[:5] if len(years) > 5 else years)

platforms = sorted(df['Platform'].unique())
selected_platforms = st.sidebar.multiselect("Платформа", platforms, default=platforms[:5] if len(platforms) > 5 else platforms)

genres = sorted(df['Genre'].unique())
selected_genres = st.sidebar.multiselect("Жанр", genres, default=genres[:5] if len(genres) > 5 else genres)

min_sales = float(df['Global_Sales'].min())
max_sales = float(df['Global_Sales'].max())
sales_range = st.sidebar.slider(
    "Глобальные продажи (млн копий)",
    min_value=min_sales,
    max_value=max_sales,
    value=(min_sales, max_sales)
)

filtered_df = df.copy()
if selected_years:
    filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
if selected_platforms:
    filtered_df = filtered_df[filtered_df['Platform'].isin(selected_platforms)]
if selected_genres:
    filtered_df = filtered_df[filtered_df['Genre'].isin(selected_genres)]
filtered_df = filtered_df[(filtered_df['Global_Sales'] >= sales_range[0]) & (filtered_df['Global_Sales'] <= sales_range[1])]

st.sidebar.markdown("---")
st.sidebar.metric("Отфильтровано записей", len(filtered_df))
st.sidebar.metric("Всего игр", len(df))

# ============================================================
# 4. КЛЮЧЕВЫЕ МЕТРИКИ
# ============================================================
st.header("Ключевые метрики")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Всего игр", f"{len(df):,}")
with col2:
    st.metric("Общие продажи", f"{df['Global_Sales'].sum():.1f}M")
with col3:
    st.metric("Средние продажи", f"{df['Global_Sales'].mean():.2f}M")
with col4:
    st.metric("Платформ", df['Platform'].nunique())
with col5:
    st.metric("Жанров", df['Genre'].nunique())

# ============================================================
# 5. ПРЕДСКАЗАНИЕ ПРОДАЖ В ЯПОНИИ
# ============================================================
st.markdown("---")
st.header("Предсказание продаж в Японии (JP_Sales)")

st.markdown("""
**Что мы предсказываем:** Продажи игры в Японии (млн копий)

**На основе каких данных:**
- Платформа (Platform)
- Жанр (Genre)
- Издатель (Publisher)
- Год выпуска (Year)
- Продажи в Северной Америке (NA_Sales)
- Продажи в Европе (EU_Sales)
- Продажи в остальном мире (Other_Sales)

**Почему это важно:** Японский рынок уникален - здесь RPG и игры Nintendo популярнее, чем в других регионах.
""")

model_type = st.selectbox(
    "Выберите модель машинного обучения",
    ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"]
)

test_size = st.slider("Размер тестовой выборки", 0.1, 0.4, 0.2, 0.05)

feature_cols = ['Platform', 'Genre', 'Publisher', 'Year', 'NA_Sales', 'EU_Sales', 'Other_Sales']
target_col = 'JP_Sales'

with st.spinner("Обучение модели..."):
    model_df = df[feature_cols + [target_col]].dropna()
    
    X = model_df[feature_cols]
    y = model_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    numeric_features = ['Year', 'NA_Sales', 'EU_Sales', 'Other_Sales']
    categorical_features = ['Platform', 'Genre', 'Publisher']
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    if model_type == "Random Forest":
        regressor = RandomForestRegressor(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1)
    elif model_type == "Gradient Boosting":
        regressor = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    elif model_type == "Ridge Regression":
        regressor = Ridge(alpha=1.0)
    else:
        regressor = LinearRegression()
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', regressor)])
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

st.subheader("Качество модели")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("R2 (обучение)", f"{train_r2:.3f}")
with col2:
    st.metric("R2 (тест)", f"{test_r2:.3f}")
with col3:
    st.metric("R2 (CV 5-fold)", f"{cv_scores.mean():.3f}")
with col4:
    st.metric("RMSE", f"{test_rmse:.3f}M")
with col5:
    st.metric("MAE", f"{test_mae:.3f}M")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_test, alpha=0.5, color='blue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Реальные продажи в Японии (млн копий)')
ax.set_ylabel('Предсказанные продажи (млн копий)')
ax.set_title(f'Предсказание продаж в Японии\nR2 = {test_r2:.3f}, RMSE = {test_rmse:.3f}M')
st.pyplot(fig)

st.subheader("Важность признаков (Random Forest / Gradient Boosting)")

rf_model = model.named_steps['regressor']
ohe = model.named_steps['preprocessor'].named_transformers_['cat']

feature_names = ohe.get_feature_names_out(categorical_features).tolist()
feature_names.extend(numeric_features)

if hasattr(rf_model, 'feature_importances_'):
    importances = rf_model.feature_importances_
else:
    # Для линейных моделей используем коэффициенты
    importances = np.abs(rf_model.coef_) if hasattr(rf_model, 'coef_') else np.zeros(len(feature_names))

importance_df = pd.DataFrame({
    'Признак': feature_names,
    'Важность': importances
}).sort_values('Важность', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(importance_df['Признак'], importance_df['Важность'], color='skyblue')
ax.set_xlabel('Важность')
ax.set_title('Топ-20 важных признаков для предсказания продаж в Японии')
ax.invert_yaxis()
st.pyplot(fig)

# ============================================================
# 6. ИНТЕРАКТИВНОЕ ПРЕДСКАЗАНИЕ
# ============================================================
st.subheader("Предсказать продажи для новой игры")
st.markdown("Введите параметры игры, чтобы получить предсказание продаж в Японии")

col1, col2 = st.columns(2)

with col1:
    pred_platform = st.selectbox("Платформа", sorted(df['Platform'].unique()))
    pred_genre = st.selectbox("Жанр", sorted(df['Genre'].unique()))
    pred_publisher = st.selectbox("Издатель", sorted(df['Publisher'].unique()))

with col2:
    pred_year = st.number_input("Год выпуска", min_value=1980, max_value=2030, value=2024)
    pred_na_sales = st.number_input("Продажи в Северной Америке (млн)", min_value=0.0, value=1.0, step=0.1)
    pred_eu_sales = st.number_input("Продажи в Европе (млн)", min_value=0.0, value=0.8, step=0.1)
    pred_other_sales = st.number_input("Продажи в остальном мире (млн)", min_value=0.0, value=0.3, step=0.1)

if st.button("Предсказать продажи в Японии", type="primary"):
    input_data = pd.DataFrame({
        'Platform': [pred_platform],
        'Genre': [pred_genre],
        'Publisher': [pred_publisher],
        'Year': [pred_year],
        'NA_Sales': [pred_na_sales],
        'EU_Sales': [pred_eu_sales],
        'Other_Sales': [pred_other_sales]
    })
    
    prediction = model.predict(input_data)[0]
    
    st.markdown("---")
    
    # Показываем факторы, влияющие на предсказание
    st.success(f"### Прогноз продаж в Японии: {prediction:.2f} млн копий")
    
    # Анализ факторов
    st.subheader("Анализ факторов, влияющих на прогноз")
    
    factors = []
    if pred_genre == 'RPG':
        factors.append("RPG жанр значительно увеличивает продажи в Японии")
    if pred_publisher == 'Nintendo':
        factors.append("Издатель Nintendo значительно увеличивает продажи в Японии")
    if pred_platform in ['Nintendo Switch', '3DS']:
        factors.append(f"Платформа {pred_platform} популярна в Японии")
    if pred_genre == 'Shooter':
        factors.append("Шутеры менее популярны в Японии")
    
    for factor in factors:
        st.write(f"- {factor}")
    
    st.info(f"""
    **Информация о модели:**
    - Доверительный интервал: +-{test_rmse * 1.96:.2f} млн копий
    - Качество модели (R2): {test_r2:.3f}
    - Cross-validation R2: {cv_scores.mean():.3f}
    - Модель обучена на {len(y_test)} играх
    """)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    regions = ['Северная Америка', 'Европа', 'Япония (прогноз)', 'Остальной мир']
    values = [pred_na_sales, pred_eu_sales, prediction, pred_other_sales]
    colors = ['blue', 'green', 'red', 'orange']
    
    bars = ax.bar(regions, values, color=colors)
    ax.set_ylabel('Продажи (млн копий)')
    ax.set_title('Сравнение продаж по регионам')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig)

# ============================================================
# 7. ТОП ИГРЫ
# ============================================================
st.markdown("---")
st.header("Топ-20 самых продаваемых игр")

top_games = filtered_df.nlargest(20, 'Global_Sales')[
    ['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales', 'JP_Sales']
]
top_games.columns = ['Название', 'Платформа', 'Год', 'Жанр', 'Издатель', 'Продажи (мир)', 'Продажи (Япония)']
st.dataframe(top_games, use_container_width=True)

# ============================================================
# 8. ЭКСПОРТ ДАННЫХ
# ============================================================
st.markdown("---")
st.header("Экспорт данных")

export_format = st.selectbox("Формат экспорта", ["CSV", "Excel", "JSON"])

all_columns = filtered_df.columns.tolist()
selected_columns = st.multiselect("Выберите колонки для экспорта", all_columns, default=['Name', 'Platform', 'Year', 'Genre', 'Global_Sales', 'JP_Sales'])

if st.button("Скачать отфильтрованные данные"):
    export_df = filtered_df[selected_columns] if selected_columns else filtered_df
    
    if export_format == "CSV":
        csv = export_df.to_csv(index=False)
        st.download_button("Скачать CSV", csv, "filtered_data.csv", "text/csv")
    elif export_format == "Excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False)
        st.download_button("Скачать Excel", output.getvalue(), "filtered_data.xlsx", 
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        json_str = export_df.to_json(orient='records', indent=2, force_ascii=False)
        st.download_button("Скачать JSON", json_str, "filtered_data.json", "application/json")

# ============================================================
# 9. О ПРИЛОЖЕНИИ
# ============================================================
st.markdown("---")
with st.expander("О приложении"):
    st.markdown("""
    **Анализ и предсказание продаж видеоигр**
    
    **Что мы предсказываем:**
    - Целевая переменная: JP_Sales (продажи в Японии в млн копий)
    
    **Почему Япония:** Японский рынок уникален - здесь RPG игры и Nintendo популярнее,
    чем в других регионах. Это делает задачу интересной и реалистичной.
    
    **Используемые признаки:**
    - Платформа (Platform)
    - Жанр (Genre)
    - Издатель (Publisher)
    - Год выпуска (Year)
    - Продажи в Северной Америке (NA_Sales)
    - Продажи в Европе (EU_Sales)
    - Продажи в остальном мире (Other_Sales)
    
    **Модели машинного обучения:**
    - Random Forest - учитывает нелинейные зависимости
    - Gradient Boosting - ансамблевый метод
    - Linear Regression - линейная модель для сравнения
    - Ridge Regression - линейная с регуляризацией
    
    **Метрики качества:**
    - R2 (коэффициент детерминации) - чем ближе к 1, тем лучше
    - RMSE (среднеквадратичная ошибка) - в млн копий
    - MAE (средняя абсолютная ошибка) - в млн копий
    - Cross-validation R2 - для проверки стабильности
    
    **Технологии:** Streamlit, Pandas, Scikit-learn, Matplotlib
    """)

st.markdown("---")
st.caption("Панель управления создана с помощью Streamlit | Предсказание продаж в Японии | Данные по видеоиграм")
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
