import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
def inject_soft_minimal_css():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FAFAFA !important;
        font-family: 'Inter', sans-serif !important;
        color: #1A1A1A !important;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid rgba(0,0,0,0.08);
    }

    h1, h2, h3 {
        font-weight: 600 !important;
        color: #0F0F0F !important;
        letter-spacing: -0.3px;
    }
    h1 { font-size: 30px !important; }
    h2 { font-size: 24px !important; }
    h3 { font-size: 18px !important; }

    .stMarkdown, .stText, .stWrite {
        font-size: 14.5px !important;
        font-weight: 400 !important;
        line-height: 1.55;
    }

    .stButton > button, button[data-baseweb="button"] {
        background-color: #3A86FF !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 10px 18px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #2F6FCE !important;
        transform: translateY(-1px);
    }

    input, textarea, select, .stTextInput > div > div > input {
        background: #FFFFFF !important;
        border-radius: 8px !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        padding: 8px 10px !important;
        color: #1A1A1A !important;
    }

    .stDataFrame, .stDataFrame table {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        font-size: 13px !important;
    }

    .block-container {
        padding-top: 2rem !important;
    }

    #MainMenu, footer { visibility: hidden !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_soft_minimal_css()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression

# Set up Streamlit page
st.set_page_config(page_title="Smart Expense Tracker", layout="wide")
st.title("Reefa")

# Currency variable
CURRENCY = "PKR"

import pickle
import os

def save_ml_model(model, vectorizer, filename="saved_ml_model.pkl"):
    """Save ML model and vectorizer to file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
        return True, f"Model saved to {filename}"
    except Exception as e:
        return False, f"Error saving model: {e}"

def load_ml_model(filename="saved_ml_model.pkl"):
    """Load ML model and vectorizer from file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return data['model'], data['vectorizer'], True
        return None, None, False
    except Exception as e:
        return None, None, False

# Load ML training data for expense categorization
@st.cache_data
def train_model():
    try:
        df_train = pd.read_csv("category_train.csv")
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df_train["Description"])
        y = df_train["Category"]
        model = MultinomialNB()
        
        # ADD TRAIN-TEST SPLIT FOR ACCURACY CALCULATION
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store accuracy for later display
        if 'model_accuracy' not in st.session_state:
            st.session_state.model_accuracy = accuracy
        
        return model, vectorizer, accuracy
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None, None, 0.0

ml_model, vectorizer, model_accuracy = train_model()

#Model Graphics in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("ML Model Info")

if st.sidebar.button("Show Model Accuracy"):
    if ml_model:
        st.sidebar.success(f"Category Model Accuracy: {model_accuracy:.1%}")
        st.sidebar.info(f"Trained on {len(pd.read_csv('category_train.csv'))} examples")
    else:
        st.sidebar.warning("Model not loaded. Check category_train.csv")

# Add in sidebar after ML Model Info section
st.sidebar.markdown("---")
st.sidebar.subheader(" Model Management")

col_save, col_load = st.sidebar.columns(2)
with col_save:
    if st.button(" Save Model"):
        if ml_model and vectorizer:
            success, message = save_ml_model(ml_model, vectorizer)
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
        else:
            st.sidebar.warning("No model to save")

with col_load:
    if st.button("Load Model"):
        loaded_model, loaded_vectorizer, success = load_ml_model()
        if success:
            ml_model = loaded_model
            vectorizer = loaded_vectorizer
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.warning("No saved model found")

# Session state initialization
if "income" not in st.session_state:
    st.session_state.income = pd.DataFrame(columns=["Date", "Category", "Amount"])
if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=["Date", "Category", "Amount"])

# --- Sidebar: Add New Expense ---
st.sidebar.header("Add New Expense")

# Initialize predicted_category
predicted_category = "Others"

# Description + ML prediction for expenses
expense_description = st.sidebar.text_input("Description (e.g., Office Lunch Contribution)", key="expense_desc")
if expense_description and ml_model and vectorizer:
    X_test = vectorizer.transform([expense_description])
    predicted_category = ml_model.predict(X_test)[0]
    st.sidebar.markdown(f"**Predicted Category:** `{predicted_category}`")
else:
    predicted_category = "Others"

# Category dropdown for expenses (with auto-selection based on ML prediction)
expense_categories = ["Food", "Transport", "Utilities", "Entertainment", "Health", "Others"]
expense_category_index = expense_categories.index(predicted_category) if predicted_category in expense_categories else 5

expense_category = st.sidebar.selectbox(
    "Select Category",
    expense_categories,
    index=expense_category_index,
    key="expense_category"
)

# Amount & Date for expenses
expense_amount = st.sidebar.number_input("Amount", min_value=0.0, format="%.2f", step=0.01, key="expense_amount")
expense_date = st.sidebar.date_input("Date", value=datetime.today(), key="expense_date")

if st.sidebar.button("Add Expense", key="add_expense"):
    new_expense = pd.DataFrame({
        "Date": [pd.to_datetime(expense_date)],
        "Category": [expense_category],
        "Amount": [expense_amount]
    })
    st.session_state.expenses = pd.concat([st.session_state.expenses, new_expense], ignore_index=True)
    st.success(" Expense added successfully!")
    st.rerun()

# --- Sidebar: Add New Income ---
st.sidebar.header("Add New Income")

# Simple category selection for income (no ML prediction)
income_category = st.sidebar.selectbox(
    "Select Category", 
    ["Salary", "Business Income", "Part-time Job salary", "Freelance", "Investment Return", "Others"],
    index=0,
    key="income_category"
)

# Amount & Date for income
income_amount = st.sidebar.number_input("Amount", min_value=0.0, format="%.2f", step=0.01, key="income_amount_input")
income_date = st.sidebar.date_input("Date", value=datetime.today(), key="income_date_input")

if st.sidebar.button("Add Income", key="add_income"):
    new_income = pd.DataFrame({
        "Date": [pd.to_datetime(income_date)],
        "Category": [income_category],
        "Amount": [income_amount]
    })
    st.session_state.income = pd.concat([st.session_state.income, new_income], ignore_index=True)
    st.success(" Income added successfully!")
    st.rerun()

# --- Main Dashboard ---

# Expense History Section
st.subheader("Expense History")
if not st.session_state.expenses.empty:
    st.dataframe(st.session_state.expenses)
    total_expenses = st.session_state.expenses["Amount"].sum()
    st.metric(f" Total Spent ({CURRENCY})", f"{total_expenses:.2f}")
else:
    st.info("No expenses recorded yet. Add your first expense from the sidebar.")

# --- Smart Insights ---
st.subheader(" Smart Insights")
if not st.session_state.expenses.empty:
    df = st.session_state.expenses.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["DayName"] = df["Date"].dt.day_name()

    # Top Category
    category_sums = df.groupby("Category")["Amount"].sum()
    top_category = category_sums.idxmax() if not category_sums.empty else "N/A"
    top_cat_amt = category_sums.max() if not category_sums.empty else 0
    total_amt = df["Amount"].sum()
    top_cat_pct = (top_cat_amt / total_amt) * 100 if total_amt > 0 else 0

    # Top Month
    month_sums = df.groupby("Month")["Amount"].sum()
    top_month = month_sums.idxmax() if not month_sums.empty else "N/A"
    top_month_amt = month_sums.max() if not month_sums.empty else 0

    # Weekend Spike
    df["IsWeekend"] = df["DayName"].isin(["Saturday", "Sunday"])
    weekend_spend = df[df["IsWeekend"]]["Amount"].sum()
    weekday_spend = df[~df["IsWeekend"]]["Amount"].sum()
    weekend_diff_pct = ((weekend_spend - weekday_spend) / weekday_spend) * 100 if weekday_spend > 0 else 0

    # Show Insights
    st.markdown(f"• **Most spending is on {top_category}** ({CURRENCY} {top_cat_amt:.2f}) — {top_cat_pct:.1f}% of total.")
    st.markdown(f"• **Highest spending month:** {top_month} ({CURRENCY} {top_month_amt:.2f})")
    st.markdown(f"• **Weekend spending is {abs(weekend_diff_pct):.1f}% {'higher' if weekend_diff_pct > 0 else 'lower'}** than weekdays.")
    
    # Average daily spending
    if len(df) > 0:
        avg_daily = df["Amount"].mean()
        st.markdown(f"• **Average daily spending:** {CURRENCY} {avg_daily:.2f}")
else:
    st.info("Add expenses to see smart insights.")

# --- Summary Cards ---
st.subheader(" Quick Summary")
if not st.session_state.expenses.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Spent", f"{CURRENCY} {total_expenses:.2f}")
    with col2:
        st.metric("Most Spent On", top_category, f"{CURRENCY} {top_cat_amt:.2f}")
    with col3:
        st.metric("Top Month", top_month, f"{CURRENCY} {top_month_amt:.2f}")
    with col4:
        label = "Higher" if weekend_diff_pct > 0 else "Lower"
        st.metric("Weekend vs Weekdays", f"{abs(weekend_diff_pct):.1f}%", label)
else:
    st.info("Add expenses to view summary.")

# --- Visualizations ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Expense by Category")
    if not st.session_state.expenses.empty:
        pie_data = st.session_state.expenses.groupby("Category")["Amount"].sum()
        fig1, ax1 = plt.subplots()
        ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

with col2:
    st.subheader("Monthly Expense Trend")
    if not st.session_state.expenses.empty:
        bar_df = st.session_state.expenses.copy()
        bar_df["Date"] = pd.to_datetime(bar_df["Date"], errors="coerce")
        bar_df["Month"] = bar_df["Date"].dt.to_period("M").astype(str)
        monthly = bar_df.groupby("Month")["Amount"].sum()
        st.bar_chart(monthly)

# --- Income vs Expenses Comparison ---
st.subheader("Income vs Expenses Overview")

if not st.session_state.expenses.empty and not st.session_state.income.empty:
    # Convert dates
    df_exp = st.session_state.expenses.copy()
    df_inc = st.session_state.income.copy()
    df_exp["Date"] = pd.to_datetime(df_exp["Date"], errors="coerce")
    df_inc["Date"] = pd.to_datetime(df_inc["Date"], errors="coerce")

    # Monthly totals
    df_exp["Month"] = df_exp["Date"].dt.to_period("M").astype(str)
    df_inc["Month"] = df_inc["Date"].dt.to_period("M").astype(str)

    monthly_expenses = df_exp.groupby("Month")["Amount"].sum()
    monthly_income = df_inc.groupby("Month")["Amount"].sum()

    # Combine into one DataFrame
    comparison = pd.DataFrame({
        "Income": monthly_income,
        "Expenses": monthly_expenses
    }).fillna(0)

    # Net savings
    comparison["Net Savings"] = comparison["Income"] - comparison["Expenses"]

    # Show metrics
    total_income = df_inc["Amount"].sum()
    total_expenses = df_exp["Amount"].sum()
    net_savings = total_income - total_expenses

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Income", f"{CURRENCY} {total_income:.2f}")
    with col2:
        st.metric("Total Expenses", f"{CURRENCY} {total_expenses:.2f}")
    with col3:
        st.metric("Net Savings", f"{CURRENCY} {net_savings:.2f}", delta_color="inverse")

    # Show comparison table
    st.dataframe(comparison)

    # Charts
    st.line_chart(comparison[["Income", "Expenses"]])
    st.bar_chart(comparison["Net Savings"])

    # --- ML Prediction for Next Month ---
    if len(monthly_expenses) > 1:  # Need at least 2 data points for regression
        X = np.arange(len(monthly_expenses)).reshape(-1, 1)
        y = monthly_expenses.values
        
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        next_month = np.array([[len(monthly_expenses)]])
        predicted_expenses = lr_model.predict(next_month)[0]
        
        # Calculate predicted net savings (assuming average monthly income)
        avg_monthly_income = total_income / max(1, len(monthly_income))
        predicted_net = avg_monthly_income - predicted_expenses
        
        st.subheader(" Next Month Prediction")
        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            st.metric("Predicted Expenses", f"{CURRENCY} {predicted_expenses:.2f}")
        with col_pred2:
            st.metric("Predicted Net Savings", f"{CURRENCY} {predicted_net:.2f}", 
                     delta_color="inverse" if predicted_net < 0 else "normal")

else:
    st.info("Add both income and expenses to see financial overview and predictions.")

# --- Filter & Export Expenses ---
st.subheader(" Filter & Export Expenses")
with st.expander("Filter Options"):
    if not st.session_state.expenses.empty:
        # Category filter
        filter_category = st.selectbox(
            "Filter by Category",
            options=["All"] + list(st.session_state.expenses["Category"].unique()),
            key="expense_filter_category"
        )

        # Date range filter
        min_date = st.session_state.expenses["Date"].min().date()
        max_date = st.session_state.expenses["Date"].max().date()
        
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input("Start Date", value=min_date, key="expense_start_date")
        with col_date2:
            end_date = st.date_input("End Date", value=max_date, key="expense_end_date")

        # Apply filters
        filtered_data = st.session_state.expenses.copy()
        filtered_data["Date"] = pd.to_datetime(filtered_data["Date"], errors="coerce")
        filtered_data = filtered_data[
            (filtered_data["Date"] >= pd.to_datetime(start_date)) &
            (filtered_data["Date"] <= pd.to_datetime(end_date))
        ]

        if filter_category != "All":
            filtered_data = filtered_data[filtered_data["Category"] == filter_category]

        # ADD AMOUNT RANGE FILTER
        st.markdown("**Amount Range Filter**")
        col_amount1, col_amount2 = st.columns(2)
        with col_amount1:
            # Get min amount from data or default to 0
            data_min = float(st.session_state.expenses["Amount"].min())
            min_amount = st.number_input(
                "Minimum Amount", 
                min_value=0.0, 
                value=0.0, 
                step=10.0,
                key="expense_min_amount"
            )
        with col_amount2:
            # Get max amount from data or default to 10000
            data_max = float(st.session_state.expenses["Amount"].max())
            max_amount = st.number_input(
                "Maximum Amount", 
                min_value=min_amount, 
                value=data_max,
                step=10.0,
                key="expense_max_amount"
            )

        if min_amount > 0 or max_amount < float('inf'):
            filtered_data = filtered_data[
                (filtered_data["Amount"] >= min_amount) & 
                (filtered_data["Amount"] <= max_amount)
            ]

        if not filtered_data.empty:
            st.dataframe(filtered_data)
            filtered_total = filtered_data["Amount"].sum()
            st.write(f"**Total in Selected Range:** {CURRENCY} {filtered_total:.2f}")

            # Export to CSV
            filename = f"expenses_{datetime.today().strftime('%Y%m%d_%H%M')}.csv"
            csv = filtered_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                " Download Filtered CSV", 
                csv, 
                filename, 
                "text/csv",
                key="expense_download"
            )
        else:
            st.warning("No data matches your filters.")
    else:
        st.info("No expenses available to filter.")

# --- Filter & Export Income ---
st.subheader(" Filter & Export Income")
with st.expander("Income Filter Options"):
    if not st.session_state.income.empty:
        # Category filter
        filter_category = st.selectbox(
            "Filter by Category",
            options=["All"] + list(st.session_state.income["Category"].unique()),
            key="income_filter_category"
        )

        # Date range filter
        min_date = st.session_state.income["Date"].min().date()
        max_date = st.session_state.income["Date"].max().date()
        
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input("Start Date", value=min_date, key="income_start_date")
        with col_date2:
            end_date = st.date_input("End Date", value=max_date, key="income_end_date")
        
        st.markdown("**Amount Range Filter**")
        col_income_amount1, col_income_amount2 = st.columns(2)
        with col_income_amount1:
            income_min = float(st.session_state.income["Amount"].min())
            min_income = st.number_input(
                "Minimum Amount", 
                min_value=0.0, 
                value=0.0, 
                step=50.0,
                key="income_min_amount"
            )
        with col_income_amount2:
            income_max = float(st.session_state.income["Amount"].max())
            max_income = st.number_input(
                "Maximum Amount", 
                min_value=min_income, 
                value=income_max,
                step=50.0,
                key="income_max_amount"
            )
        
        # Apply filters
        filtered_income = st.session_state.income.copy()
        filtered_income["Date"] = pd.to_datetime(filtered_income["Date"], errors="coerce")
        filtered_income = filtered_income[
            (filtered_income["Date"] >= pd.to_datetime(start_date)) &
            (filtered_income["Date"] <= pd.to_datetime(end_date))
        ]

        if filter_category != "All":
            filtered_income = filtered_income[filtered_income["Category"] == filter_category]
        
        # Apply amount range filter to income
        filtered_income = filtered_income[
            (filtered_income["Amount"] >= min_income) & 
            (filtered_income["Amount"] <= max_income)
        ]

        if not filtered_income.empty:
            st.dataframe(filtered_income)
            filtered_income_total = filtered_income["Amount"].sum()
            st.write(f"**Total Income in Selected Range:** {CURRENCY} {filtered_income_total:.2f}")

            # Export to CSV
            filename = f"income_{datetime.today().strftime('%Y%m%d_%H%M')}.csv"
            csv = filtered_income.to_csv(index=False).encode("utf-8")
            st.download_button(
                " Download Filtered Income CSV", 
                csv, 
                filename, 
                "text/csv",
                key="income_download"
            )
        else:
            st.warning("No income data matches your filters.")
    else:
        st.info("No income records available to filter.")

# --- Clear Data Option ---
st.subheader(" Data Management")
if st.button(" Clear All Data", type="secondary"):
    st.session_state.expenses = pd.DataFrame(columns=["Date", "Category", "Amount"])
    st.session_state.income = pd.DataFrame(columns=["Date", "Category", "Amount"])
    st.success("All data cleared successfully!")
    st.rerun()
