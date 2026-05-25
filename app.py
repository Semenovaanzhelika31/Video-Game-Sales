import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from io import BytesIO

# ============================================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="Video Game Sales Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Video Game Sales Analytics & Prediction Dashboard")
st.markdown("---")

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
st.sidebar.header("Data Upload")

# Загрузка файла
uploaded_file = st.sidebar.file_uploader("Upload CSV file (vgsales_cleaned.csv)", type=['csv'])

# Кнопка для загрузки примера
if st.sidebar.button("Load vgsales_cleaned.csv"):
    try:
        df = pd.read_csv('vgsales_cleaned.csv')
        st.session_state['df'] = df
        st.success("File loaded successfully")
    except FileNotFoundError:
        st.error("File 'vgsales_cleaned.csv' not found. Please upload the file.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
    st.success("File uploaded successfully")

if 'df' not in st.session_state:
    st.info("Please upload vgsales_cleaned.csv file or click 'Load vgsales_cleaned.csv'")
    st.stop()

df = st.session_state['df']

# ============================================================
# 2. ОТОБРАЖЕНИЕ ДАННЫХ
# ============================================================
st.header("Raw Data")
st.markdown(f"**Data size:** {df.shape[0]} rows, {df.shape[1]} columns")

rows_to_show = st.selectbox("Rows to display:", [5, 10, 20, 50, 100], index=1)
st.dataframe(df.head(rows_to_show), use_container_width=True)

with st.expander("Data Information"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data types:**")
        st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Type'}))
    with col2:
        st.write("**Basic statistics:**")
        st.dataframe(df.describe())

# ============================================================
# 3. ФИЛЬТРЫ
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("Filters")

# Фильтр по годам
years = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect("Year", years, default=years[:5] if len(years) > 5 else years)

# Фильтр по платформам
platforms = sorted(df['Platform'].unique())
selected_platforms = st.sidebar.multiselect("Platform", platforms, default=platforms[:5] if len(platforms) > 5 else platforms)

# Фильтр по жанрам
genres = sorted(df['Genre'].unique())
selected_genres = st.sidebar.multiselect("Genre", genres, default=genres[:5] if len(genres) > 5 else genres)

# Фильтр по продажам
min_sales = float(df['Global_Sales'].min())
max_sales = float(df['Global_Sales'].max())
sales_range = st.sidebar.slider(
    "Global Sales (million copies)",
    min_value=min_sales,
    max_value=max_sales,
    value=(min_sales, max_sales)
)

# Применение фильтров
filtered_df = df.copy()
if selected_years:
    filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
if selected_platforms:
    filtered_df = filtered_df[filtered_df['Platform'].isin(selected_platforms)]
if selected_genres:
    filtered_df = filtered_df[filtered_df['Genre'].isin(selected_genres)]
filtered_df = filtered_df[(filtered_df['Global_Sales'] >= sales_range[0]) & (filtered_df['Global_Sales'] <= sales_range[1])]

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Records", len(filtered_df))
st.sidebar.metric("Total Games", len(df))

# ============================================================
# 4. КЛЮЧЕВЫЕ МЕТРИКИ
# ============================================================
st.header("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Games", f"{len(df):,}")
with col2:
    st.metric("Total Sales", f"{df['Global_Sales'].sum():.1f}M")
with col3:
    st.metric("Avg Sales", f"{df['Global_Sales'].mean():.2f}M")
with col4:
    st.metric("Platforms", df['Platform'].nunique())
with col5:
    st.metric("Genres", df['Genre'].nunique())

# ============================================================
# 5. ПРЕДСКАЗАНИЕ ПРОДАЖ В ЯПОНИИ
# ============================================================
st.markdown("---")
st.header("Japan Sales Prediction (JP_Sales)")

st.markdown("""
This model predicts Japan sales based on:
- Platform, Genre, Publisher, Year
- Sales in other regions (NA_Sales, EU_Sales, Other_Sales)
""")

# Выбор модели
model_type = st.selectbox(
    "Select ML Model",
    ["Random Forest (recommended)", "Linear Regression"]
)

test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

# Подготовка признаков
feature_cols = ['Platform', 'Genre', 'Publisher', 'Year', 'NA_Sales', 'EU_Sales', 'Other_Sales']
target_col = 'JP_Sales'

with st.spinner("Training model..."):
    # Удаляем строки с пропусками
    model_df = df[feature_cols + [target_col]].dropna()
    
    X = model_df[feature_cols]
    y = model_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Определяем типы признаков
    numeric_features = ['Year', 'NA_Sales', 'EU_Sales', 'Other_Sales']
    categorical_features = ['Platform', 'Genre', 'Publisher']
    
    # Pipeline
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    if model_type == "Random Forest (recommended)":
        regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        regressor = LinearRegression()
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', regressor)])
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Метрики
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

# Отображение метрик
st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("R2 (train)", f"{train_r2:.3f}")
with col2:
    st.metric("R2 (test)", f"{test_r2:.3f}")
with col3:
    st.metric("RMSE", f"{test_rmse:.3f}M")
with col4:
    st.metric("MAE", f"{test_mae:.3f}M")

# График предсказаний
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_test, alpha=0.5, color='blue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Japan Sales (million)')
ax.set_ylabel('Predicted Japan Sales (million)')
ax.set_title(f'Japan Sales Prediction\nR2 = {test_r2:.3f}, RMSE = {test_rmse:.3f}M')
st.pyplot(fig)

# Важность признаков для Random Forest
if "Random Forest" in model_type:
    st.subheader("Feature Importance")
    
    rf_model = model.named_steps['regressor']
    ohe = model.named_steps['preprocessor'].named_transformers_['cat']
    
    feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names.extend(numeric_features)
    
    importances = rf_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Important Features for JP_Sales Prediction')
    ax.invert_yaxis()
    st.pyplot(fig)

# ============================================================
# 6. ИНТЕРАКТИВНОЕ ПРЕДСКАЗАНИЕ
# ============================================================
st.subheader("Predict Sales for a New Game")
st.markdown("Enter game parameters to get Japan sales prediction")

col1, col2 = st.columns(2)

with col1:
    pred_platform = st.selectbox("Platform", sorted(df['Platform'].unique()))
    pred_genre = st.selectbox("Genre", sorted(df['Genre'].unique()))
    pred_publisher = st.selectbox("Publisher", sorted(df['Publisher'].unique()))

with col2:
    pred_year = st.number_input("Release Year", min_value=1980, max_value=2030, value=2024)
    pred_na_sales = st.number_input("North America Sales (million)", min_value=0.0, value=1.0, step=0.1)
    pred_eu_sales = st.number_input("Europe Sales (million)", min_value=0.0, value=0.8, step=0.1)
    pred_other_sales = st.number_input("Other Regions Sales (million)", min_value=0.0, value=0.3, step=0.1)

if st.button("Predict Japan Sales", type="primary"):
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
    st.success(f"### Predicted Japan Sales: {prediction:.2f} million copies")
    
    st.info(f"""
    **Model Information:**
    - Confidence interval: +-{test_rmse * 1.96:.2f} million copies
    - Model R2 score: {test_r2:.3f}
    - Tested on {len(y_test)} games
    """)
    
    # Сравнение регионов
    fig, ax = plt.subplots(figsize=(8, 5))
    regions = ['NA_Sales', 'EU_Sales', 'JP_Sales (predicted)', 'Other_Sales']
    values = [pred_na_sales, pred_eu_sales, prediction, pred_other_sales]
    colors = ['blue', 'green', 'red', 'orange']
    
    ax.bar(regions, values, color=colors)
    ax.set_ylabel('Sales (million copies)')
    ax.set_title('Regional Sales Comparison')
    st.pyplot(fig)

# ============================================================
# 7. ВИЗУАЛИЗАЦИИ
# ============================================================
st.markdown("---")
st.header("Data Visualization")

chart_type = st.selectbox(
    "Select Chart Type",
    ["Top 10 Platforms", "Sales by Genre", "Trend by Year", "Regional Distribution", "Top Publishers", "Sales Distribution"]
)

# Top 10 Platforms
if chart_type == "Top 10 Platforms":
    platform_sales = filtered_df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    platform_sales.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel('Global Sales (million copies)')
    ax.set_title('Top 10 Platforms by Sales')
    ax.invert_yaxis()
    st.pyplot(fig)

# Sales by Genre
elif chart_type == "Sales by Genre":
    genre_sales = filtered_df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    genre_sales.plot(kind='bar', color='lightcoral', ax=ax)
    ax.set_xlabel('Genre')
    ax.set_ylabel('Global Sales (million copies)')
    ax.set_title('Sales by Genre')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# Trend by Year
elif chart_type == "Trend by Year":
    yearly_sales = filtered_df.groupby('Year')['Global_Sales'].sum().reset_index()
    
    fig = px.line(yearly_sales, x='Year', y='Global_Sales', 
                  title='Global Sales Trend by Year',
                  labels={'Global_Sales': 'Sales (million copies)', 'Year': 'Year'})
    st.plotly_chart(fig, use_container_width=True)

# Regional Distribution
elif chart_type == "Regional Distribution":
    regions = {
        'NA_Sales': 'North America',
        'EU_Sales': 'Europe', 
        'JP_Sales': 'Japan',
        'Other_Sales': 'Other'
    }
    
    region_sales = {regions[reg]: filtered_df[reg].sum() for reg in regions.keys()}
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(region_sales.values(), labels=region_sales.keys(), autopct='%1.1f%%', startangle=90)
    ax.set_title('Regional Sales Distribution')
    st.pyplot(fig)

# Top Publishers
elif chart_type == "Top Publishers":
    publisher_sales = filtered_df.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    publisher_sales.plot(kind='barh', color='lightgreen', ax=ax)
    ax.set_xlabel('Global Sales (million copies)')
    ax.set_title('Top 15 Publishers by Sales')
    ax.invert_yaxis()
    st.pyplot(fig)

# Sales Distribution
elif chart_type == "Sales Distribution":
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(filtered_df['Global_Sales'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Global Sales (million copies)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Global Sales')
    ax.axvline(filtered_df['Global_Sales'].median(), color='red', linestyle='--', label=f'Median: {filtered_df["Global_Sales"].median():.2f}M')
    ax.legend()
    st.pyplot(fig)

# ============================================================
# 8. ТОП ИГРЫ
# ============================================================
st.markdown("---")
st.header("Top 20 Best Selling Games")

top_games = filtered_df.nlargest(20, 'Global_Sales')[
    ['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales', 'JP_Sales']
]
st.dataframe(top_games, use_container_width=True)

# ============================================================
# 9. ЭКСПОРТ ДАННЫХ
# ============================================================
st.markdown("---")
st.header("Export Data")

export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])

all_columns = filtered_df.columns.tolist()
selected_columns = st.multiselect("Select columns to export", all_columns, default=['Name', 'Platform', 'Year', 'Genre', 'Global_Sales', 'JP_Sales'])

if st.button("Download Filtered Data"):
    export_df = filtered_df[selected_columns] if selected_columns else filtered_df
    
    if export_format == "CSV":
        csv = export_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")
    elif export_format == "Excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False)
        st.download_button("Download Excel", output.getvalue(), "filtered_data.xlsx", 
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        json_str = export_df.to_json(orient='records', indent=2)
        st.download_button("Download JSON", json_str, "filtered_data.json", "application/json")

# ============================================================
# 10. О ПРИЛОЖЕНИИ
# ============================================================
st.markdown("---")
with st.expander("About this app"):
    st.markdown("""
    **Video Game Sales Analytics & Prediction Dashboard**
    
    **Features:**
    - Data analysis with filtering and visualization
    - Japan sales prediction using Random Forest / Linear Regression
    - Interactive prediction for new games
    - Data export in CSV, Excel, JSON formats
    
    **Target Variable:** JP_Sales (Japan sales in million copies)
    
    **Features used for prediction:**
    - Platform, Genre, Publisher (categorical)
    - Year, NA_Sales, EU_Sales, Other_Sales (numerical)
    
    **Technologies:**
    - Streamlit, Pandas, Scikit-learn
    - Plotly, Matplotlib, Seaborn
    """)

st.markdown("---")
st.caption("Dashboard created with Streamlit | Japan Sales Prediction | Video Game Data")
