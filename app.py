import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("vgsales.csv")

    if "Rank" in df.columns:
        df = df.drop(columns=["Rank"])

    df = df.dropna(subset=["Year", "Global_Sales"])
    df["Year"] = df["Year"].astype(int)

    needed_cols = [
        "Name", "Platform", "Year", "Genre", "Publisher",
        "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"
    ]
    df = df[needed_cols].copy()

    for col in ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"])
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
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Колонка", 0: "Тип"}))
    with col2:
        st.write("**Основная статистика:**")
        st.dataframe(df.describe())

# ============================================================
# 3. ФИЛЬТРЫ
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("Фильтры")

years = sorted(df["Year"].unique())
selected_years = st.sidebar.multiselect("Год выпуска", years, default=years[:5] if len(years) > 5 else years)

platforms = sorted(df["Platform"].unique())
selected_platforms = st.sidebar.multiselect("Платформа", platforms, default=platforms[:5] if len(platforms) > 5 else platforms)

genres = sorted(df["Genre"].unique())
selected_genres = st.sidebar.multiselect("Жанр", genres, default=genres[:5] if len(genres) > 5 else genres)

min_sales = float(df["Global_Sales"].min())
max_sales = float(df["Global_Sales"].max())
sales_range = st.sidebar.slider(
    "Глобальные продажи (млн копий)",
    min_value=min_sales,
    max_value=max_sales,
    value=(min_sales, max_sales)
)

filtered_df = df.copy()
if selected_years:
    filtered_df = filtered_df[filtered_df["Year"].isin(selected_years)]
if selected_platforms:
    filtered_df = filtered_df[filtered_df["Platform"].isin(selected_platforms)]
if selected_genres:
    filtered_df = filtered_df[filtered_df["Genre"].isin(selected_genres)]
filtered_df = filtered_df[
    (filtered_df["Global_Sales"] >= sales_range[0]) &
    (filtered_df["Global_Sales"] <= sales_range[1])
]

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
    st.metric("Платформ", df["Platform"].nunique())
with col5:
    st.metric("Жанров", df["Genre"].nunique())

# ============================================================
# 5. ПРЕДСКАЗАНИЕ JP_SALES
# ============================================================
st.markdown("---")
st.header("Предсказание продаж в Японии (JP_Sales)")

st.markdown("""
**Что мы предсказываем:** Продажи игры в Японии (млн копий).

**На основе каких данных:**
- Платформа
- Жанр
- Издатель
- Год выпуска
- Продажи в Северной Америке
- Продажи в Европе
- Продажи в остальном мире

**Почему это важно:** Японский рынок уникален — здесь RPG и игры Nintendo популярнее, чем в других регионах.
""")

model_type = st.selectbox(
    "Выберите модель машинного обучения",
    ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"]
)

test_size = st.slider("Размер тестовой выборки", 0.1, 0.4, 0.2, 0.05)

feature_cols = ["Platform", "Genre", "Publisher", "Year", "NA_Sales", "EU_Sales", "Other_Sales"]
target_col = "JP_Sales"

with st.spinner("Обучение модели..."):
    model_df = df[feature_cols + [target_col]].dropna()

    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    numeric_features = ["Year", "NA_Sales", "EU_Sales", "Other_Sales"]
    categorical_features = ["Platform", "Genre", "Publisher"]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    if model_type == "Random Forest":
        regressor = RandomForestRegressor(
            n_estimators=500,
            max_depth=25,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "Gradient Boosting":
        regressor = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
    elif model_type == "Ridge Regression":
        regressor = Ridge(alpha=1.0)
    else:
        regressor = LinearRegression()

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

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
ax.scatter(y_test, y_pred_test, alpha=0.5, color="blue")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax.set_xlabel("Реальные продажи в Японии (млн копий)")
ax.set_ylabel("Предсказанные продажи (млн копий)")
ax.set_title(f"Предсказание продаж в Японии\nR2 = {test_r2:.3f}, RMSE = {test_rmse:.3f}M")
st.pyplot(fig)

# ============================================================
# 6. ВАЖНОСТЬ ПРИЗНАКОВ
# ============================================================
st.subheader("Важность признаков")

cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
feature_names.extend(numeric_features)

rf_model = model.named_steps["regressor"]

if hasattr(rf_model, "feature_importances_"):
    importances = rf_model.feature_importances_
elif hasattr(rf_model, "coef_"):
    importances = np.abs(rf_model.coef_)
else:
    importances = np.zeros(len(feature_names))

importance_df = pd.DataFrame({
    "Признак": feature_names,
    "Важность": importances
}).sort_values("Важность", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(importance_df["Признак"], importance_df["Важность"], color="skyblue")
ax.set_xlabel("Важность")
ax.set_title("Топ-20 важных признаков для предсказания JP_Sales")
ax.invert_yaxis()
st.pyplot(fig)

# ============================================================
# 7. ИНТЕРАКТИВНОЕ ПРЕДСКАЗАНИЕ
# ============================================================
st.subheader("Предсказать продажи для новой игры")
st.markdown("Введите параметры игры, чтобы получить предсказание продаж в Японии.")

col1, col2 = st.columns(2)

with col1:
    pred_platform = st.selectbox("Платформа", sorted(df["Platform"].unique()))
    pred_genre = st.selectbox("Жанр", sorted(df["Genre"].unique()))
    pred_publisher = st.selectbox("Издатель", sorted(df["Publisher"].unique()))

with col2:
    pred_year = st.number_input("Год выпуска", min_value=1980, max_value=2030, value=2024)
    pred_na_sales = st.number_input("Продажи в Северной Америке (млн)", min_value=0.0, value=1.0, step=0.1)
    pred_eu_sales = st.number_input("Продажи в Европе (млн)", min_value=0.0, value=0.8, step=0.1)
    pred_other_sales = st.number_input("Продажи в остальном мире (млн)", min_value=0.0, value=0.3, step=0.1)

if st.button("Предсказать продажи в Японии", type="primary"):
    input_data = pd.DataFrame({
        "Platform": [pred_platform],
        "Genre": [pred_genre],
        "Publisher": [pred_publisher],
        "Year": [pred_year],
        "NA_Sales": [pred_na_sales],
        "EU_Sales": [pred_eu_sales],
        "Other_Sales": [pred_other_sales]
    })

    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.success(f"### Прогноз продаж в Японии: {prediction:.2f} млн копий")

    st.subheader("Анализ факторов, влияющих на прогноз")
    factors = []

    if pred_genre == "RPG":
        factors.append("RPG-жанр обычно увеличивает продажи в Японии.")
    if pred_publisher == "Nintendo":
        factors.append("Nintendo часто усиливает продажи на японском рынке.")
    if pred_platform in ["Nintendo Switch", "3DS"]:
        factors.append(f"Платформа {pred_platform} популярна в Японии.")
    if pred_genre == "Shooter":
        factors.append("Шутеры обычно слабее продаются в Японии.")

    if factors:
        for factor in factors:
            st.write(f"- {factor}")
    else:
        st.write("- Явных дополнительных рыночных факторов не выделено.")

    st.info(f"""
    **Информация о модели:**
    - Доверительный интервал: +- {test_rmse * 1.96:.2f} млн копий
    - Качество модели (R2): {test_r2:.3f}
    - Cross-validation R2: {cv_scores.mean():.3f}
    - Модель обучена на {len(y_test)} играх
    """)

    fig, ax = plt.subplots(figsize=(8, 5))
    regions = ["Северная Америка", "Европа", "Япония (прогноз)", "Остальной мир"]
    values = [pred_na_sales, pred_eu_sales, prediction, pred_other_sales]
    colors = ["blue", "green", "red", "orange"]

    bars = ax.bar(regions, values, color=colors)
    ax.set_ylabel("Продажи (млн копий)")
    ax.set_title("Сравнение продаж по регионам")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    st.pyplot(fig)

# ============================================================
# 8. ТОП-20 ИГР
# ============================================================
st.markdown("---")
st.header("Топ-20 самых продаваемых игр")

top_games = filtered_df.nlargest(20, "Global_Sales")[
    ["Name", "Platform", "Year", "Genre", "Publisher", "Global_Sales", "JP_Sales"]
].copy()

top_games.columns = ["Название", "Платформа", "Год", "Жанр", "Издатель", "Продажи (мир)", "Продажи (Япония)"]
st.dataframe(top_games, use_container_width=True)

# ============================================================
# 9. ЭКСПОРТ ДАННЫХ
# ============================================================
st.markdown("---")
st.header("Экспорт данных")

export_format = st.selectbox("Формат экспорта", ["CSV", "Excel", "JSON"])

all_columns = filtered_df.columns.tolist()
selected_columns = st.multiselect(
    "Выберите колонки для экспорта",
    all_columns,
    default=["Name", "Platform", "Year", "Genre", "Global_Sales", "JP_Sales"]
)

if st.button("Скачать отфильтрованные данные"):
    export_df = filtered_df[selected_columns] if selected_columns else filtered_df

    if export_format == "CSV":
        csv = export_df.to_csv(index=False)
        st.download_button("Скачать CSV", csv, "filtered_data.csv", "text/csv")
    elif export_format == "Excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False)
        st.download_button(
            "Скачать Excel",
            output.getvalue(),
            "filtered_data.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        json_str = export_df.to_json(orient="records", indent=2, force_ascii=False)
        st.download_button("Скачать JSON", json_str, "filtered_data.json", "application/json")

# ============================================================
# 10. О ПРИЛОЖЕНИИ
# ============================================================
st.markdown("---")
with st.expander("О приложении"):
    st.markdown("""
    **Анализ и предсказание продаж видеоигр**

    **Что мы предсказываем:**
    - Целевая переменная: JP_Sales

    **Почему Япония:**
    Японский рынок уникален — здесь RPG и Nintendo популярнее, чем в других регионах.

    **Используемые признаки:**
    - Платформа
    - Жанр
    - Издатель
    - Год выпуска
    - Продажи в Северной Америке
    - Продажи в Европе
    - Продажи в остальном мире

    **Модели машинного обучения:**
    - Random Forest
    - Gradient Boosting
    - Linear Regression
    - Ridge Regression

    **Метрики качества:**
    - R2
    - RMSE
    - MAE
    - Cross-validation R2

    **Технологии:** Streamlit, Pandas, Scikit-learn, Matplotlib
    """)

st.markdown("---")
st.caption("Панель управления создана с помощью Streamlit | Предсказание продаж в Японии | Данные по видеоиграм")
