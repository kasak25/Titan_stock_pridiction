import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    /* Cards */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e3e7ed;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0px 3px 10px rgba(0, 0, 0, 0.05);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
        font-size: 16px;
        font-weight: 600;
    }

    /* Info boxes */
    .info-box {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e3e7ed;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* Prediction result */
    .prediction-value {
        font-size: 42px;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-top: 10px;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #777777;
        padding-top: 40px;
        padding-bottom: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "TITAN.csv")
MODEL_FILE = os.path.join(BASE_DIR, "stock_model.pkl")


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATA_FILE):

    st.error("❌ TITAN.csv was not found.")

    st.info(
        "Please keep TITAN.csv in the same folder as app.py."
    )

    st.stop()


try:

    df = pd.read_csv(DATA_FILE)

except Exception as e:

    st.error(f"❌ Error loading dataset: {e}")

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"❌ Missing columns: {missing_columns}"
    )

    st.write(
        "Available columns:"
    )

    st.write(
        df.columns.tolist()
    )

    st.stop()


# ============================================================
# CONVERT DATE
# ============================================================

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=numeric_columns
).copy()


# ============================================================
# SORT DATA
# ============================================================

if "Date" in df.columns:

    df = df.sort_values(
        "Date"
    )

    df.reset_index(
        drop=True,
        inplace=True
    )


# ============================================================
# EXTRA FEATURES
# ============================================================

df["Daily Return"] = (
    df["Close"].pct_change() * 100
)

df["MA20"] = (
    df["Close"].rolling(20).mean()
)

df["Price Change"] = (
    df["Close"].diff()
)

df["Price Change %"] = (
    df["Close"].pct_change() * 100
)


# ============================================================
# LOAD MODEL
# ============================================================

model = None

if os.path.exists(MODEL_FILE):

    try:

        model = joblib.load(
            MODEL_FILE
        )

    except Exception as e:

        st.warning(
            f"⚠️ Model could not be loaded: {e}"
        )

else:

    st.warning(
        "⚠️ stock_model.pkl was not found."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📈 Stock Prediction")

st.sidebar.write(
    "Machine Learning Stock Market Analysis"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🔮 Prediction",
        "🤖 Model Performance",
        "📁 Dataset",
        "ℹ️ About Project"
    ]
)


# ============================================================
# DATE FILTER
# ============================================================

filtered_df = df.copy()

if "Date" in df.columns:

    valid_dates = df["Date"].dropna()

    if len(valid_dates) > 0:

        st.sidebar.subheader(
            "📅 Date Filter"
        )

        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        start_date = st.sidebar.date_input(
            "Start Date",
            min_value=min_date,
            max_value=max_date,
            value=min_date
        )

        end_date = st.sidebar.date_input(
            "End Date",
            min_value=min_date,
            max_value=max_date,
            value=max_date
        )

        if start_date <= end_date:

            filtered_df = df[
                (df["Date"].dt.date >= start_date)
                &
                (df["Date"].dt.date <= end_date)
            ].copy()


# ============================================================
# PAGE 1 - DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("📈 Stock Market Prediction")

    st.write(
        "Analyze historical stock market data, "
        "understand market trends and predict "
        "closing prices using Machine Learning."
    )

    st.divider()


    if filtered_df.empty:

        st.warning(
            "No data available for the selected date range."
        )

        st.stop()


    # --------------------------------------------------------
    # LATEST VALUES
    # --------------------------------------------------------

    latest = filtered_df.iloc[-1]

    latest_close = float(
        latest["Close"]
    )

    latest_open = float(
        latest["Open"]
    )

    latest_high = float(
        latest["High"]
    )

    latest_low = float(
        latest["Low"]
    )

    latest_volume = float(
        latest["Volume"]
    )


    if len(filtered_df) > 1:

        previous_close = float(
            filtered_df["Close"].iloc[-2]
        )

    else:

        previous_close = latest_close


    price_change = (
        latest_close -
        previous_close
    )


    if previous_close != 0:

        price_change_percent = (
            price_change /
            previous_close
        ) * 100

    else:

        price_change_percent = 0


    # --------------------------------------------------------
    # MARKET OVERVIEW
    # --------------------------------------------------------

    st.header("📊 Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Latest Close",
            f"₹{latest_close:,.2f}"
        )

    with col2:

        st.metric(
            "Highest Price",
            f"₹{filtered_df['High'].max():,.2f}"
        )

    with col3:

        st.metric(
            "Lowest Price",
            f"₹{filtered_df['Low'].min():,.2f}"
        )

    with col4:

        st.metric(
            "Average Close",
            f"₹{filtered_df['Close'].mean():,.2f}"
        )


    # --------------------------------------------------------
    # MARKET MOVEMENT
    # --------------------------------------------------------

    st.header("📈 Latest Market Movement")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Open",
            f"₹{latest_open:,.2f}"
        )

    with col2:

        st.metric(
            "High",
            f"₹{latest_high:,.2f}"
        )

    with col3:

        st.metric(
            "Low",
            f"₹{latest_low:,.2f}"
        )

    with col4:

        st.metric(
            "Price Change",
            f"₹{price_change:,.2f}",
            f"{price_change_percent:.2f}%"
        )


    # --------------------------------------------------------
    # CLOSING PRICE
    # --------------------------------------------------------

    st.header("📈 Closing Price Trend")

    if "Date" in filtered_df.columns:

        chart_data = (
            filtered_df
            .set_index("Date")
            [["Close"]]
        )

        st.line_chart(
            chart_data,
            use_container_width=True
        )

    else:

        st.line_chart(
            filtered_df[["Close"]],
            use_container_width=True
        )


    # --------------------------------------------------------
    # OPEN VS CLOSE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "💹 Open vs Close"
        )

        if "Date" in filtered_df.columns:

            chart_data = (
                filtered_df
                .set_index("Date")
                [["Open", "Close"]]
            )

        else:

            chart_data = filtered_df[
                ["Open", "Close"]
            ]

        st.line_chart(
            chart_data
        )


    with col2:

        st.subheader(
            "📦 Trading Volume"
        )

        if "Date" in filtered_df.columns:

            volume_data = (
                filtered_df
                .set_index("Date")
                [["Volume"]]
            )

        else:

            volume_data = filtered_df[
                ["Volume"]
            ]

        st.bar_chart(
            volume_data
        )


    # --------------------------------------------------------
    # MARKET SUMMARY
    # --------------------------------------------------------

    st.header("💡 Market Summary")

    if price_change > 0:

        st.success(
            "📈 The latest closing price is higher "
            "than the previous closing price."
        )

    elif price_change < 0:

        st.warning(
            "📉 The latest closing price is lower "
            "than the previous closing price."
        )

    else:

        st.info(
            "➡️ The latest closing price is unchanged."
        )


    st.info(
        f"Latest trading volume: "
        f"{latest_volume:,.0f}"
    )


    # --------------------------------------------------------
    # PROJECT WORKFLOW
    # --------------------------------------------------------

    st.header("🧠 Project Workflow")

    workflow = [
        "📁 Dataset Collection",
        "🧹 Data Preprocessing",
        "📊 Exploratory Data Analysis",
        "🎯 Feature Selection",
        "🤖 Machine Learning Model Training",
        "📈 Model Evaluation",
        "🔮 Stock Price Prediction"
    ]

    for i, step in enumerate(
        workflow,
        start=1
    ):

        st.write(
            f"**{i}. {step}**"
        )


# ============================================================
# PAGE 2 - DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.title("📊 Data Analysis")

    st.write(
        "Explore stock prices, trading volume, "
        "returns and relationships between variables."
    )

    st.divider()


    if filtered_df.empty:

        st.warning(
            "No data available."
        )

        st.stop()


    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    st.header("📋 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            f"{len(filtered_df):,}"
        )

    with col2:

        st.metric(
            "Columns",
            len(filtered_df.columns)
        )

    with col3:

        st.metric(
            "Missing Values",
            int(
                filtered_df.isnull().sum().sum()
            )
        )

    with col4:

        st.metric(
            "Duplicate Rows",
            int(
                filtered_df.duplicated().sum()
            )
        )


    # --------------------------------------------------------
    # CLOSING PRICE
    # --------------------------------------------------------

    st.header("📈 Closing Price")

    if "Date" in filtered_df.columns:

        chart = (
            filtered_df
            .set_index("Date")
            [["Close"]]
        )

    else:

        chart = filtered_df[
            ["Close"]
        ]

    st.line_chart(
        chart
    )


    # --------------------------------------------------------
    # MOVING AVERAGE
    # --------------------------------------------------------

    st.header("📊 Closing Price vs 20-Day Moving Average")

    ma_chart = filtered_df[
        ["Close", "MA20"]
    ].dropna()

    if "Date" in filtered_df.columns:

        ma_chart.index = (
            filtered_df.loc[
                ma_chart.index,
                "Date"
            ]
        )

    st.line_chart(
        ma_chart
    )


    # --------------------------------------------------------
    # HIGH VS LOW
    # --------------------------------------------------------

    st.header("🔺 High vs Low")

    high_low = filtered_df[
        ["High", "Low"]
    ]

    if "Date" in filtered_df.columns:

        high_low = high_low.copy()

        high_low.index = (
            filtered_df["Date"]
        )

    st.line_chart(
        high_low
    )


    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    st.header("📦 Trading Volume")

    volume_chart = filtered_df[
        ["Volume"]
    ]

    if "Date" in filtered_df.columns:

        volume_chart = volume_chart.copy()

        volume_chart.index = (
            filtered_df["Date"]
        )

    st.bar_chart(
        volume_chart
    )


    # --------------------------------------------------------
    # DAILY RETURN
    # --------------------------------------------------------

    st.header("📉 Daily Returns")

    return_chart = filtered_df[
        ["Daily Return"]
    ].dropna()

    if "Date" in filtered_df.columns:

        return_chart.index = (
            filtered_df.loc[
                return_chart.index,
                "Date"
            ]
        )

    st.line_chart(
        return_chart
    )


    # --------------------------------------------------------
    # PRICE DISTRIBUTION
    # --------------------------------------------------------

    st.header("📊 Closing Price Distribution")

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.hist(
        filtered_df["Close"].dropna(),
        bins=30
    )

    ax.set_title(
        "Distribution of Closing Prices"
    )

    ax.set_xlabel(
        "Closing Price"
    )

    ax.set_ylabel(
        "Frequency"
    )

    st.pyplot(
        fig
    )

    plt.close(fig)


    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.header("🔗 Feature Correlation")

    correlation_columns = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Close"
    ]

    correlation = filtered_df[
        correlation_columns
    ].corr()


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    im = ax.imshow(
        correlation,
        interpolation="nearest"
    )

    ax.set_xticks(
        range(
            len(correlation_columns)
        )
    )

    ax.set_yticks(
        range(
            len(correlation_columns)
        )
    )

    ax.set_xticklabels(
        correlation_columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        correlation_columns
    )

    for i in range(
        len(correlation_columns)
    ):

        for j in range(
            len(correlation_columns)
        ):

            ax.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title(
        "Correlation Matrix"
    )

    fig.colorbar(
        im,
        ax=ax
    )

    st.pyplot(
        fig
    )

    plt.close(fig)


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    st.header("📋 Statistical Summary")

    st.dataframe(
        filtered_df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ].describe(),
        use_container_width=True
    )


# ============================================================
# PAGE 3 - PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.title("🔮 Stock Price Prediction")

    st.write(
        "Enter the market values and use the trained "
        "Machine Learning model to predict the closing price."
    )

    st.divider()


    if model is None:

        st.error(
            "❌ stock_model.pkl is not available."
        )

        st.info(
            "Place stock_model.pkl in the same "
            "folder as app.py."
        )

        st.stop()


    latest = df.iloc[-1]


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    st.header("📥 Enter Stock Information")

    col1, col2 = st.columns(2)


    with col1:

        open_price = st.number_input(
            "Open Price",
            min_value=0.0,
            value=float(
                latest["Open"]
            ),
            step=0.01
        )

        high_price = st.number_input(
            "High Price",
            min_value=0.0,
            value=float(
                latest["High"]
            ),
            step=0.01
        )


    with col2:

        low_price = st.number_input(
            "Low Price",
            min_value=0.0,
            value=float(
                latest["Low"]
            ),
            step=0.01
        )

        volume = st.number_input(
            "Trading Volume",
            min_value=0,
            value=int(
                latest["Volume"]
            ),
            step=1000
        )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if high_price < low_price:

        st.warning(
            "⚠️ High price should not be lower than Low price."
        )


    if open_price < low_price:

        st.warning(
            "⚠️ Open price is below the Low price."
        )


    if open_price > high_price:

        st.warning(
            "⚠️ Open price is above the High price."
        )


    st.write("")


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Closing Price"
    ):

        try:

            input_data = pd.DataFrame({
                "Open": [
                    open_price
                ],
                "High": [
                    high_price
                ],
                "Low": [
                    low_price
                ],
                "Volume": [
                    volume
                ]
            })


            prediction = model.predict(
                input_data
            )


            predicted_price = float(
                prediction[0]
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.success(
                "✅ Prediction generated successfully!"
            )

            st.subheader(
                "🎯 Predicted Closing Price"
            )

            st.markdown(
                f'<div class="prediction-value">₹{predicted_price:,.2f}</div>',
                unsafe_allow_html=True
            )


            st.divider()


            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📋 Input Summary"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Open",
                    f"₹{open_price:,.2f}"
                )

            with col2:

                st.metric(
                    "High",
                    f"₹{high_price:,.2f}"
                )

            with col3:

                st.metric(
                    "Low",
                    f"₹{low_price:,.2f}"
                )

            with col4:

                st.metric(
                    "Volume",
                    f"{volume:,.0f}"
                )


            # ------------------------------------------------
            # COMPARISON
            # ------------------------------------------------

            actual_close = float(
                df["Close"].iloc[-1]
            )

            difference = (
                predicted_price -
                actual_close
            )


            if actual_close != 0:

                difference_percent = (
                    difference /
                    actual_close
                ) * 100

            else:

                difference_percent = 0


            st.subheader(
                "📊 Prediction Comparison"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Latest Actual Close",
                    f"₹{actual_close:,.2f}"
                )

            with col2:

                st.metric(
                    "Predicted Close",
                    f"₹{predicted_price:,.2f}"
                )

            with col3:

                st.metric(
                    "Difference",
                    f"₹{difference:,.2f}",
                    f"{difference_percent:.2f}%"
                )


            if predicted_price > actual_close:

                st.info(
                    "📈 The model prediction is higher "
                    "than the latest actual closing price."
                )

            elif predicted_price < actual_close:

                st.warning(
                    "📉 The model prediction is lower "
                    "than the latest actual closing price."
                )

            else:

                st.success(
                    "➡️ The model prediction matches "
                    "the latest actual closing price."
                )


        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


# ============================================================
# PAGE 4 - MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")
    st.write("Complete visualization of the regression model's performance metrics.")
    st.divider()

    if model is None:
        st.error("❌ Model not found. Keep stock_model.pkl beside app.py.")
        st.stop()

    # Use the SAME four features used to train stock_model.pkl.
    feature_columns = ["Open", "High", "Low", "Volume"]
    target_column = "Close"

    evaluation_df = df.dropna(subset=feature_columns + [target_column]).copy()
    X = evaluation_df[feature_columns]
    y = evaluation_df[target_column]

    # Recreate the same 80/20 evaluation setup used by the notebook.
    # random_state is fixed so the frontend gives reproducible metrics.
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    try:
        # The saved LinearRegression model is already trained.
        test_predictions = model.predict(X_test_eval)
        train_predictions = model.predict(X_train_eval)
        test_predictions = np.asarray(test_predictions).flatten()
        train_predictions = np.asarray(train_predictions).flatten()
    except Exception as e:
        st.error(f"❌ Could not generate model predictions: {e}")
        st.stop()

    # -------------------------
    # Metrics
    # -------------------------
    mae = mean_absolute_error(y_test_eval, test_predictions)
    mse = mean_squared_error(y_test_eval, test_predictions)
    rmse = np.sqrt(mse)
    rss = np.sum((y_test_eval.values - test_predictions) ** 2)
    r2 = r2_score(y_test_eval, test_predictions)
    train_r2 = r2_score(y_train_eval, train_predictions)
    residuals = y_test_eval.values - test_predictions
    absolute_errors = np.abs(residuals)

    st.header("📊 Evaluation Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{mae:.6f}")
    c2.metric("MSE", f"{mse:.6f}")
    c3.metric("RMSE", f"{rmse:.6f}")
    c4.metric("RSS", f"{rss:.6f}")
    c5.metric("R² Score", f"{r2:.6f}")

    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "RSS", "R²"],
        "Value": [mae, mse, rmse, rss, r2]
    })

    # -------------------------
    # 1. Actual vs Predicted
    # -------------------------
    st.header("1️⃣ Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y_test_eval, test_predictions, alpha=0.65)
    lo = min(y_test_eval.min(), test_predictions.min())
    hi = max(y_test_eval.max(), test_predictions.max())
    ax.plot([lo, hi], [lo, hi], linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Close Price")
    ax.set_ylabel("Predicted Close Price")
    ax.set_title("Actual vs Predicted Values")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # 2. Actual and Predicted over samples
    # -------------------------
    st.header("2️⃣ Actual vs Predicted Across Test Samples")
    plot_df = pd.DataFrame({
        "Actual": y_test_eval.values,
        "Predicted": test_predictions
    })
    if "Date" in evaluation_df.columns:
        dates = evaluation_df.loc[X_test_eval.index, "Date"]
        plot_df.index = dates
        plot_df = plot_df.sort_index()
    st.line_chart(plot_df, use_container_width=True)

    # -------------------------
    # 3. Residual plot
    # -------------------------
    st.header("3️⃣ Residual Plot")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(test_predictions, residuals, alpha=0.65)
    ax.axhline(0, linewidth=2)
    ax.set_xlabel("Predicted Close Price")
    ax.set_ylabel("Residual Error")
    ax.set_title("Residual Plot")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # 4. Residual distribution
    # -------------------------
    st.header("4️⃣ Residual Distribution")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.75)
    ax.axvline(0, linewidth=2)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Residuals")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # 5. Absolute error
    # -------------------------
    st.header("5️⃣ Absolute Prediction Error")
    error_plot = pd.DataFrame({"Absolute Error": absolute_errors})
    if "Date" in evaluation_df.columns:
        error_plot.index = evaluation_df.loc[X_test_eval.index, "Date"]
        error_plot = error_plot.sort_index()
    st.line_chart(error_plot, use_container_width=True)

    # -------------------------
    # 6. Error metrics graph
    # -------------------------
    st.header("6️⃣ Error Metrics Comparison")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(["MAE", "MSE", "RMSE", "RSS"], [mae, mse, rmse, rss])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Regression Error Metrics")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # 7. R² graph / overfitting check
    # -------------------------
    st.header("7️⃣ Training vs Testing R²")
    r2_df = pd.DataFrame({"R²": [train_r2, r2]}, index=["Training", "Testing"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(r2_df.index, r2_df["R²"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² Score")
    ax.set_title("Training vs Testing R²")
    ax.grid(axis="y", alpha=0.3)
    for i, value in enumerate(r2_df["R²"]):
        ax.text(i, value + 0.02, f"{value:.4f}", ha="center")
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # 8. Cross-validation
    # -------------------------
    st.header("8️⃣ 5-Fold Cross-Validation R²")
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i}" for i in range(1, 6)],
            "R²": cv_scores
        })
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(cv_df["Fold"], cv_df["R²"])
        ax.axhline(cv_scores.mean(), linewidth=2,
                   label=f"Mean = {cv_scores.mean():.4f}")
        ax.set_ylabel("R² Score")
        ax.set_title("5-Fold Cross-Validation R²")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Cross-validation could not be calculated: {e}")

    # -------------------------
    # 9. Correlation between actual and prediction errors
    # -------------------------
    st.header("9️⃣ Prediction Error Analysis")
    error_analysis = pd.DataFrame({
        "Actual": y_test_eval.values,
        "Predicted": test_predictions,
        "Residual": residuals,
        "Absolute Error": absolute_errors
    })
    st.dataframe(error_analysis.head(100), use_container_width=True, hide_index=True)

    # -------------------------
    # 10. Metrics table
    # -------------------------
    st.header("🔟 Complete Metrics Table")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # -------------------------
    # Download results
    # -------------------------
    results_download = error_analysis.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Prediction Results",
        results_download,
        "prediction_results.csv",
        "text/csv"
    )

    st.info(
        "The Streamlit frontend is connected directly to stock_model.pkl and TITAN.csv. "
        "The saved LinearRegression model uses Open, High, Low and Volume as inputs and Close as the target, "
        "matching the project notebook."
    )


# ============================================================
# PAGE 5 - DATASET
# ============================================================

elif page == "📁 Dataset":

    st.title("📁 Dataset")

    st.write(
        "Explore the complete stock market dataset."
    )

    st.divider()


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.header(
        "📊 Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Columns",
            len(df.columns)
        )

    with col3:

        st.metric(
            "Missing Values",
            int(
                df.isnull().sum().sum()
            )
        )

    with col4:

        st.metric(
            "Duplicates",
            int(
                df.duplicated().sum()
            )
        )


    # --------------------------------------------------------
    # DATE RANGE
    # --------------------------------------------------------

    if "Date" in df.columns:

        st.header(
            "📅 Date Range"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "First Date",
                str(
                    df["Date"].min().date()
                )
            )

        with col2:

            st.metric(
                "Last Date",
                str(
                    df["Date"].max().date()
                )
            )


    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    st.header(
        "🔤 Column Information"
    )

    information = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(
                df[col].dtype
            )
            for col in df.columns
        ],
        "Missing Values": [
            int(
                df[col].isnull().sum()
            )
            for col in df.columns
        ]
    })


    st.dataframe(
        information,
        use_container_width=True
    )


    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.header(
        "📄 Dataset Preview"
    )

    number_of_rows = st.slider(
        "Rows to display",
        10,
        min(500, len(df)),
        min(50, len(df))
    )


    st.dataframe(
        df.head(number_of_rows),
        use_container_width=True,
        height=500
    )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    st.header(
        "📋 Statistical Summary"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ============================================================
# PAGE 6 - ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title(
        "ℹ️ About the Project"
    )

    st.write(
        "Stock Market Prediction using Machine Learning"
    )

    st.divider()


    # --------------------------------------------------------
    # PROBLEM STATEMENT
    # --------------------------------------------------------

    st.header(
        "📌 Problem Statement"
    )

    st.info(
        "Stock prices change continuously based on "
        "different market conditions and historical trends. "
        "This project analyzes historical stock market data "
        "and uses Machine Learning to predict the closing price."
    )


    # --------------------------------------------------------
    # OBJECTIVES
    # --------------------------------------------------------

    st.header(
        "🎯 Project Objectives"
    )

    objectives = [
        "Analyze historical stock market data.",
        "Perform data preprocessing.",
        "Understand stock price trends.",
        "Analyze relationships between variables.",
        "Select suitable Machine Learning features.",
        "Train a Machine Learning model.",
        "Evaluate model performance.",
        "Predict future closing prices.",
        "Build an interactive frontend using Streamlit."
    ]

    for objective in objectives:

        st.write(
            f"✓ {objective}"
        )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.header(
        "📥 Input Features"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Feature 1",
            "Open"
        )

    with col2:

        st.metric(
            "Feature 2",
            "High"
        )

    with col3:

        st.metric(
            "Feature 3",
            "Low"
        )

    with col4:

        st.metric(
            "Feature 4",
            "Volume"
        )


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    st.header(
        "🎯 Target Variable"
    )

    st.success(
        "Close — The model predicts the closing price."
    )


    # --------------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------------

    st.header(
        "🛠️ Technologies Used"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "🐍 Python"
        )

        st.write(
            "Pandas"
        )

        st.write(
            "NumPy"
        )

        st.write(
            "Matplotlib"
        )


    with col2:

        st.subheader(
            "🤖 Machine Learning"
        )

        st.write(
            "Scikit-learn"
        )

        st.write(
            "Joblib"
        )


    with col3:

        st.subheader(
            "🌐 Frontend"
        )

        st.write(
            "Streamlit"
        )

        st.write(
            "Custom CSS"
        )


    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.header(
        "🔄 Project Workflow"
    )

    workflow = [
        "Dataset",
        "Data Preprocessing",
        "Exploratory Data Analysis",
        "Feature Selection",
        "Model Training",
        "Model Evaluation",
        "Prediction"
    ]

    for i, step in enumerate(
        workflow,
        1
    ):

        st.write(
            f"**{i}.** {step}"
        )


    # --------------------------------------------------------
    # HOW PREDICTION WORKS
    # --------------------------------------------------------

    st.header(
        "🔮 How Prediction Works"
    )

    st.write(
        """
        1. User enters Open, High, Low and Volume.

        2. The values are converted into a DataFrame.

        3. The trained Machine Learning model receives
           these features.

        4. The model predicts the Close price.

        5. The predicted closing price is displayed.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📈 Stock Market Prediction | "
    "Machine Learning Project | "
    "Python + Streamlit"
)
