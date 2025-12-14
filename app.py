import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("No-Code ML Pipeline Builder")

st.markdown("""
This app allows you to build and run a simple ML pipeline without writing code. 
Follow the steps below to upload data, preprocess, split, select a model, and view results.
""")

# Step 1: Dataset Upload
st.header("1. Dataset Upload")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File uploaded successfully!")
        st.subheader("Dataset Information")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
        st.write("Column Names:")
        st.write(", ".join(df.columns))
        
        # Preview the data
        if st.checkbox("Show data preview"):
            st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {str(e)}. Please upload a valid CSV or Excel file.")

# Proceed only if data is loaded
if df is not None:
    # Select target and features
    st.subheader("Select Target Column")
    st.markdown("Choose the column that represents the target variable (what you want to predict). The rest will be features.")
    target_col = st.selectbox("Target Column", options=df.columns)
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Automatically select numeric features for scaling (since scalers work on numerics)
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for preprocessing. Scaling will be skipped.")
    else:
        X = X[numeric_cols]  # Use only numeric features
    
    # Step 2: Data Preprocessing
    st.header("2. Data Preprocessing")
    st.markdown("Select a preprocessing method to apply to your numeric features.")
    scaler_option = st.selectbox("Preprocessing Method", options=["None", "Standardization (StandardScaler)", "Normalization (MinMaxScaler)"])
    
    if scaler_option != "None":
        if scaler_option == "Standardization (StandardScaler)":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        st.success(f"{scaler_option} applied successfully!")
    
    # Step 3: Train-Test Split
    st.header("3. Train-Test Split")
    st.markdown("Split your dataset into training and testing sets.")
    split_ratio = st.slider("Test set size (percentage)", min_value=10, max_value=50, value=20, step=5)
    test_size = split_ratio / 100.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.success("Dataset split successfully!")
    st.write(f"Training set: {X_train.shape[0]} rows")
    st.write(f"Testing set: {X_test.shape[0]} rows")
    
    # Step 4: Model Selection
    st.header("4. Model Selection")
    st.markdown("Choose one model to train.")
    model_option = st.selectbox("Model", options=["Logistic Regression", "Decision Tree Classifier"])
    
    # Step 5: Train and Display Results
    st.header("5. Train Model and View Results")
    if st.button("Run Pipeline"):
        with st.spinner("Training model..."):
            if model_option == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = DecisionTreeClassifier(random_state=42)
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success("Model trained successfully!")
                st.subheader("Model Results")
                st.write("**Execution Status:** Success")
                st.write(f"**Accuracy:** {accuracy:.4f}")
                
                # Visualization: Confusion Matrix
                if len(y.unique()) <= 10:  # Limit to reasonable number of classes
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                else:
                    st.info("Confusion matrix not displayed due to too many classes.")
            except Exception as e:
                st.error(f"Error training model: {str(e)}. Ensure your target is suitable for classification and data is properly preprocessed.")
