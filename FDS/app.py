import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import io

# 🎨 Custom theme and styling
st.set_page_config(
    page_title="📱 Mobile Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 💅 Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        border-radius: 10px;
        background-color: #1E1E1E;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        transform: translateY(-2px);
    }
    .footer {
        color: #FFFFFF;
        text-align: center;
        padding: 1.5rem;
        background-color: #2C2C2C;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-content {
        background-color: #2C2C2C;
        color: #FFFFFF;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .header-content h4 {
        color: #FF4B4B;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .header-content p {
        color: #E0E0E0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# 📌 File paths with environment variable support
train_path = os.getenv("TRAIN_PATH", "/home/xlegion/Documents/FDS/train.csv")
test_path = os.getenv("TEST_PATH", "/home/xlegion/Documents/FDS/test.csv")

# 🔍 Enhanced file check
if not os.path.exists(train_path) or not os.path.exists(test_path):
    st.error("❌ Dataset files not found!", icon="🚨")
    st.info("Please ensure the following files exist:")
    st.code(f"Training data: {train_path}\nTest data: {test_path}")
    st.stop()

# 🛠 Enhanced data loading with progress
@st.cache_data(show_spinner=False)
def load_data():
    with st.spinner("Loading datasets..."):
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    return df_train, df_test

# 🏆 Enhanced model training with progress tracking
@st.cache_resource
def train_model(df_train):
    with st.spinner("Training model... This may take a moment."):
        X = df_train.drop(columns=['price_range'])
        y = df_train['price_range']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        return grid.best_estimator_, scaler, X_val_scaled, y_val

# 🎯 Modern UI Components
st.title("📱 Mobile Price Predictor")
st.markdown("""
    <div class="header-content">
        <h4>AI-Powered Mobile Price Prediction</h4>
        <p>Experience the future of mobile price prediction with our cutting-edge AI model. Get instant, accurate price range estimates based on comprehensive device specifications and market analysis.</p>
    </div>
""", unsafe_allow_html=True)

# 🚀 Load data with status
with st.spinner("Initializing..."):
    df_train, df_test = load_data()
    model, scaler, X_val, y_val = train_model(df_train)

# 📊 Interactive Feature Input
st.header("🔍 Enter Mobile Features")
tabs = st.tabs(["Single Prediction 📱", "Batch Prediction 📊"])

feature_names = list(df_train.columns.drop('price_range'))

with tabs[0]:
    cols = st.columns(4)
    inputs = {}

    for i, feature in enumerate(feature_names):
        with cols[i % 4]:
            if feature in ['blue', 'dual_sim', 'four_g', 'five_g', 'touch_screen', 'wifi']:
                inputs[feature] = st.toggle(
                    f"🔹 {feature.replace('_', ' ').title()}", 
                    key=feature
                )
            else:
                min_val = int(df_train[feature].min())
                max_val = int(df_train[feature].max())

                # 🔥 Modern Feature Ranges
                if feature == "battery_power":
                    max_val = 5000
                elif feature == "ram":
                    max_val = 8000
                elif feature == "n_cores":
                    max_val = 12
                elif feature == "m_dep":
                    min_val = 0.0
                    max_val = 1.0
                    value = 0.5
                    inputs[feature] = st.slider(
                        f"📊 {feature.replace('_', ' ').title()}", 
                        min_val, max_val, value,
                        step=0.1,
                        key=feature,
                        help=f"Typical range: {min_val} - {max_val}"
                    )
                    continue

                value = int(df_train[feature].median())
                
                inputs[feature] = st.slider(
                    f"📊 {feature.replace('_', ' ').title()}", 
                    min_val, max_val, value,
                    key=feature,
                    help=f"Typical range: {min_val} - {max_val}"
                )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Predict Price Range", use_container_width=True):
            with st.spinner("Analyzing features..."):
                try:
                    input_array = np.array([list(inputs.values())]).astype(float)
                    scaled_input = scaler.transform(input_array)
                    prediction = model.predict(scaled_input)[0]
                    proba = model.predict_proba(scaled_input)[0]
                    
                    # 📊 Animated result display
                    price_ranges = {
                        0: ("Low- 💰", "#28a745"),
                        1: ("Mid-Range 💎", "#ffc107"),
                        2: ("High-End ✨", "#17a2b8"),
                        3: ("Premium 👑", "#dc3545")
                    }
                    
                    result_col1, result_col2 = st.columns([2, 1])
                    with result_col1:
                        st.success(f"🎯 Predicted Category: {price_ranges[prediction][0]}")
                        
                        # Add confidence bar chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Low-', 'Mid-Range', 'High-End', 'Premium'],
                                y=proba * 100,
                                marker_color=['#28a745', '#ffc107', '#17a2b8', '#dc3545']
                            )
                        ])
                        fig.update_layout(
                            title="Prediction Confidence",
                            yaxis_title="Confidence (%)",
                            height=300,
                            plot_bgcolor='#2C2C2C',
                            paper_bgcolor='#2C2C2C',
                            font=dict(color='#FFFFFF')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}", icon="🚨")

with tabs[1]:
    uploaded_file = st.file_uploader(
        "📂 Upload CSV file:",
        type=["csv"],
        help="Upload a CSV file with the same features as the training data"
    )
    
    if uploaded_file:
        try:
            test_df = pd.read_csv(uploaded_file)
            if not set(feature_names).issubset(test_df.columns):
                st.error("❌ Missing required features in uploaded file!", icon="⚠️")
                st.info("Required features:", feature_names)
            else:
                with st.spinner("Processing batch predictions..."):
                    X_test = test_df[feature_names]
                    scaled_test = scaler.transform(X_test)
                    predictions = model.predict(scaled_test)
                    test_df['predicted_price_range'] = predictions

                    st.success("✅ Predictions completed successfully!")
                    
                    # Interactive results table
                    st.dataframe(
                        test_df.style.background_gradient(subset=['predicted_price_range']),
                        use_container_width=True
                    )

                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = test_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download CSV",
                            data=csv,
                            file_name='mobile_price_predictions.csv',
                            mime='text/csv'
                        )
                    with col2:
                        # Create Excel file in memory
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            test_df.to_excel(writer, index=False)
                        excel_buffer.seek(0)
                        
                        st.download_button(
                            "📥 Download Excel",
                            data=excel_buffer,
                            file_name='mobile_price_predictions.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}", icon="🚨")

# 📈 Enhanced Model Performance Section
st.header("📈 Model Performance")

# Create tabs for different metrics
metric_tabs = st.tabs(["Accuracy 🎯", "Confusion Matrix 🔢", "Detailed Report 📊"])

with metric_tabs[0]:
    accuracy = accuracy_score(y_val, model.predict(X_val))
    st.metric(
        "Validation Accuracy",
        f"{accuracy:.2%}",
        delta_color="normal"
    )

with metric_tabs[1]:
    fig = px.imshow(
        confusion_matrix(y_val, model.predict(X_val)),
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=['Low-Range', 'Mid-Range', 'High-End', 'Premium'],
        y=['Low-Range', 'Mid-Range', 'High-End', 'Premium'],
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(
        plot_bgcolor='#2C2C2C',
        paper_bgcolor='#2C2C2C',
        font=dict(color='#FFFFFF')
    )
    st.plotly_chart(fig, use_container_width=True)

with metric_tabs[2]:
    report = classification_report(y_val, model.predict(X_val))
    st.code(report)

# ℹ️ Enhanced About Section
st.header("ℹ️ About This App")
with st.expander("Learn More About the Mobile Price Predictor"):
    st.markdown("""
    ### 🤖 Advanced AI Technology
    This app leverages a **Support Vector Machine (SVM) classifier** with optimized hyperparameters to predict mobile phone price ranges based on specifications.
    
    ### 📱 Feature Analysis
    We analyze various phone specifications including:
    - 🔋 Battery capacity (up to **5000 mAh**)
    - 🔵 Wireless connectivity (Bluetooth, WiFi)
    - 💾 RAM specifications (up to **8GB**)
    - 📱 Display metrics
    - 📷 Camera capabilities
    - 📶 Network support (4G & 5G)
    - 🏆 Processor details (up to **12 cores**)
    
    ### 🛠 Technical Stack
    - **Frontend:** Streamlit with custom styling
    - **Backend:** Python with scikit-learn
    - **ML Model:** Support Vector Machine with GridSearchCV optimization
    - **Data Processing:** Pandas & NumPy
    
    """)

# Add footer with light text on dark background
st.markdown("""
    <div class="footer">
        <p style="color: #FFFFFF; font-size: 1.1rem;">Made with ❤️ by Trinity | © 2025 | Empowering Mobile Price Predictions</p>
    </div>
""", unsafe_allow_html=True)
