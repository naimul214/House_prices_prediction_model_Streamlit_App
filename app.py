"""
Ontario House Price Predictor - Streamlit Application

A machine learning web application for predicting house prices in Ontario
using Linear Regression and Random Forest models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data import load_data, clean_data, engineer_features, prepare_features, load_encoders
from src.train import train_all_models, load_model, get_feature_importance
from src.predict import (
    load_model_and_encoders, prepare_single_input, predict_price,
    predict_with_confidence, batch_predict, validate_input, get_feature_names
)
from src.utils import (
    validate_csv_upload, format_currency, get_area_statistics,
    create_summary_statistics, create_results_dataframe
)

# Page configuration
st.set_page_config(
    page_title="Ontario House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prepare_data():
    """Load and prepare the dataset (cached)."""
    df = load_data("properties.csv")
    df = clean_data(df)
    df = engineer_features(df)
    return df


@st.cache_resource
def load_models_and_encoders():
    """Load trained models and encoders (cached)."""
    try:
        model_lr, encoders_lr = load_model_and_encoders("models/model_lr.joblib")
        model_rf, encoders_rf = load_model_and_encoders("models/model_rf.joblib")
        return model_lr, model_rf, encoders_lr
    except FileNotFoundError:
        return None, None, None


def train_models_ui(use_scaling, lr_params, rf_params, random_state):
    """Train models with UI feedback."""
    with st.spinner("🔄 Training models... This may take a minute."):
        try:
            models, metrics, X_train, X_test, y_train, y_test = train_all_models(
                data_path="properties.csv",
                random_state=random_state,
                use_scaling=use_scaling,
                lr_params=lr_params,
                rf_params=rf_params,
                save_models_flag=True,
                models_dir="models"
            )
            st.success("✅ Models trained and saved successfully!")
            return models, metrics, True
        except Exception as e:
            st.error(f"❌ Error training models: {str(e)}")
            return None, None, False


def sidebar_controls():
    """Render sidebar controls and return configuration."""
    st.sidebar.title("⚙️ Model Configuration")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        options=["Linear Regression", "Random Forest"],
        help="Choose the machine learning model for predictions"
    )
    
    st.sidebar.divider()
    
    # Hyperparameters based on model choice
    if model_choice == "Linear Regression":
        st.sidebar.subheader("Linear Regression Settings")
        use_regularization = st.checkbox(
            "Enable Regularization (Ridge)",
            value=False,
            help="Apply L2 regularization to prevent overfitting"
        )
        lr_params = {'use_regularization': use_regularization}
        rf_params = None
    else:
        st.sidebar.subheader("Random Forest Settings")
        n_estimators = st.slider(
            "Number of Trees",
            min_value=50,
            max_value=300,
            value=100,
            step=10,
            help="Number of decision trees in the forest"
        )
        max_depth = st.slider(
            "Maximum Tree Depth",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Maximum depth of each decision tree"
        )
        rf_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        lr_params = None
    
    st.sidebar.divider()
    
    # Additional settings
    use_scaling = st.sidebar.checkbox(
        "Enable Feature Scaling",
        value=True,
        help="Standardize features to have mean=0 and std=1"
    )
    
    random_state = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        help="Set random seed for reproducibility"
    )
    
    st.sidebar.divider()
    
    # Model retraining
    if st.sidebar.button("🔄 Retrain Models", help="Train new models with current settings"):
        st.session_state.retrain_models = True
    
    return model_choice, use_scaling, lr_params, rf_params, random_state


def predict_tab(model, encoders, df_reference):
    """Render the prediction tab."""
    st.header("🔮 Make Predictions")
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        options=["Upload CSV", "Manual Input", "Use Sample Data"],
        horizontal=True
    )
    
    predictions_df = None
    
    if input_method == "Upload CSV":
        st.subheader("📤 Upload Property Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV with columns: Address, AreaName, lat, lng"
        )
        
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                
                # Validate CSV
                is_valid, error_msg, _ = validate_csv_upload(df_input)
                if not is_valid:
                    st.error(f"⚠️ {error_msg}")
                    return
                
                st.success(f"✅ Loaded {len(df_input)} properties")
                st.dataframe(df_input.head(), use_container_width=True)
                
                if st.button("🎯 Predict Prices", type="primary"):
                    with st.spinner("Predicting..."):
                        predictions_df = batch_predict(model, encoders, df_input, df_reference)
                        st.session_state.predictions = predictions_df
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    elif input_method == "Manual Input":
        st.subheader("✏️ Enter Property Details")
        
        # Get unique areas from reference data
        unique_areas = sorted(df_reference['AreaName'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            address = st.text_input(
                "Address",
                value="123 Main Street Toronto, ON",
                help="Full property address"
            )
            area_name = st.selectbox(
                "Area Name",
                options=unique_areas,
                help="Neighborhood or area"
            )
        
        with col2:
            lat = st.number_input(
                "Latitude",
                min_value=41.0,
                max_value=57.0,
                value=43.65,
                step=0.01,
                format="%.4f",
                help="Property latitude (41-57°N)"
            )
            lng = st.number_input(
                "Longitude",
                min_value=-95.0,
                max_value=-74.0,
                value=-79.38,
                step=0.01,
                format="%.4f",
                help="Property longitude (-95 to -74°W)"
            )
        
        if st.button("🎯 Predict Price", type="primary"):
            # Validate input
            is_valid, error_msg = validate_input(address, area_name, lat, lng)
            
            if not is_valid:
                st.error(f"⚠️ {error_msg}")
            else:
                with st.spinner("Predicting..."):
                    try:
                        X_input = prepare_single_input(
                            address, area_name, lat, lng, encoders, df_reference
                        )
                        
                        pred, lower, upper = predict_with_confidence(model, X_input)
                        
                        st.success("✅ Prediction Complete!")
                        
                        # Display result prominently
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Price", format_currency(pred[0]))
                        with col2:
                            st.metric("Lower Bound (95% CI)", format_currency(lower[0]))
                        with col3:
                            st.metric("Upper Bound (95% CI)", format_currency(upper[0]))
                        
                        # Area statistics
                        area_stats = get_area_statistics(df_reference, area_name)
                        if area_stats['count'] > 0:
                            st.info(f"""
                            📊 **{area_name} Statistics:**
                            - Average Price: {format_currency(area_stats['avg_price'])}
                            - Median Price: {format_currency(area_stats['median_price'])}
                            - Properties: {area_stats['count']}
                            """)
                        
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
    
    else:  # Use Sample Data
        st.subheader("📋 Sample Data")
        sample_df = df_reference.sample(n=10, random_state=42)[['Address', 'AreaName', 'lat', 'lng', 'Price ($)']]
        st.dataframe(sample_df, use_container_width=True)
        
        if st.button("🎯 Predict Sample Prices", type="primary"):
            with st.spinner("Predicting..."):
                predictions_df = batch_predict(model, encoders, sample_df, df_reference)
                st.session_state.predictions = predictions_df
    
    # Display predictions if available
    if 'predictions' in st.session_state or predictions_df is not None:
        if predictions_df is None:
            predictions_df = st.session_state.predictions
        
        st.divider()
        st.subheader("📊 Prediction Results")
        
        # Create display dataframe
        display_cols = ['Address', 'AreaName', 'Predicted_Price']
        if 'Price ($)' in predictions_df.columns:
            display_cols.extend(['Price ($)', 'Prediction_Error', 'Absolute_Error'])
        
        display_df = predictions_df[display_cols].copy()
        display_df['Predicted_Price'] = display_df['Predicted_Price'].apply(format_currency)
        
        if 'Price ($)' in predictions_df.columns:
            display_df['Price ($)'] = display_df['Price ($)'].apply(format_currency)
            display_df['Prediction_Error'] = display_df['Prediction_Error'].apply(format_currency)
            display_df['Absolute_Error'] = display_df['Absolute_Error'].apply(format_currency)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="house_price_predictions.csv",
            mime="text/csv",
            help="Download predictions as CSV file"
        )


def metrics_tab(model_choice):
    """Render the metrics and performance tab."""
    st.header("📈 Model Performance Metrics")
    
    # Load test results if available
    try:
        # Retrain to get metrics (in production, these would be saved)
        with st.spinner("Loading model metrics..."):
            df = load_and_prepare_data()
            from src.data import prepare_features, split_data
            from sklearn.preprocessing import StandardScaler
            
            X, encoders = prepare_features(df, fit_encoders=True, scaler=StandardScaler())
            y = df['Price ($)']
            X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)
            
            model_lr, model_rf, _ = load_models_and_encoders()
            
            if model_lr and model_rf:
                from src.train import evaluate_model
                metrics_lr = evaluate_model(model_lr, X_train, y_train, X_test, y_test)
                metrics_rf = evaluate_model(model_rf, X_train, y_train, X_test, y_test)
                
                # Display metrics
                st.subheader("Model Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Linear Regression")
                    met_col1, met_col2, met_col3 = st.columns(3)
                    with met_col1:
                        st.metric("RMSE", f"${metrics_lr['test_rmse']:,.0f}")
                    with met_col2:
                        st.metric("MAE", f"${metrics_lr['test_mae']:,.0f}")
                    with met_col3:
                        st.metric("R² Score", f"{metrics_lr['test_r2']:.4f}")
                
                with col2:
                    st.markdown("### Random Forest")
                    met_col1, met_col2, met_col3 = st.columns(3)
                    with met_col1:
                        st.metric("RMSE", f"${metrics_rf['test_rmse']:,.0f}")
                    with met_col2:
                        st.metric("MAE", f"${metrics_rf['test_mae']:,.0f}")
                    with met_col3:
                        st.metric("R² Score", f"{metrics_rf['test_r2']:.4f}")
                
                st.divider()
                
                # Metrics explanation
                with st.expander("ℹ️ Understanding the Metrics"):
                    st.markdown("""
                    - **RMSE (Root Mean Squared Error)**: Average prediction error in dollars. Lower is better.
                    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices.
                    - **R² Score**: Proportion of variance explained by the model (0-1). Higher is better.
                    """)
                
                # Dataset info
                st.subheader("📊 Dataset Information")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Training Samples", len(X_train))
                with col2:
                    st.metric("Test Samples", len(X_test))
                with col3:
                    st.metric("Features", X_train.shape[1])
                with col4:
                    st.metric("Test Split", "20%")
                
    except Exception as e:
        st.error(f"❌ Error loading metrics: {str(e)}")
        st.info("💡 Try retraining the models using the sidebar controls.")


def visualizations_tab(model_choice):
    """Render the visualizations tab."""
    st.header("📊 Data Visualizations")
    
    df = load_and_prepare_data()
    
    # Visualization selector
    viz_type = st.selectbox(
        "Select Visualization",
        [
            "Price Distribution by Area",
            "Geographic Price Map",
            "Feature Importance (Random Forest)",
            "Price vs Distance from Downtown",
            "Top 10 Most Expensive Areas"
        ]
    )
    
    if viz_type == "Price Distribution by Area":
        # Get top areas by property count
        top_areas = df['AreaName'].value_counts().head(15).index
        df_filtered = df[df['AreaName'].isin(top_areas)]
        
        fig = px.box(
            df_filtered,
            x='AreaName',
            y='Price ($)',
            title="House Price Distribution by Area (Top 15 Areas)",
            labels={'AreaName': 'Area Name', 'Price ($)': 'Price ($)'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Geographic Price Map":
        # Sample for performance
        df_sample = df.sample(n=min(1000, len(df)), random_state=42)
        
        fig = px.scatter_mapbox(
            df_sample,
            lat='lat',
            lon='lng',
            color='Price ($)',
            size='Price ($)',
            hover_data=['Address', 'AreaName'],
            title="House Prices Across Ontario",
            zoom=8,
            height=600,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Feature Importance (Random Forest)":
        if model_choice != "Random Forest":
            st.warning("⚠️ Feature importance is only available for Random Forest model.")
            st.info("💡 Please select 'Random Forest' in the sidebar to view this visualization.")
        else:
            try:
                model_rf, _ = load_model_and_encoders("models/model_rf.joblib")
                feature_names = get_feature_names()
                importance_df = get_feature_importance(model_rf, feature_names)
                
                if importance_df is not None:
                    fig = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Feature Importance in Random Forest Model",
                        labels={'importance': 'Importance Score', 'feature': 'Feature'}
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Unable to extract feature importance.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif viz_type == "Price vs Distance from Downtown":
        fig = px.scatter(
            df.sample(min(1000, len(df)), random_state=42),
            x='Distance_From_Downtown',
            y='Price ($)',
            color='AreaName',
            title="House Price vs Distance from Downtown Toronto",
            labels={
                'Distance_From_Downtown': 'Distance from Downtown (degrees)',
                'Price ($)': 'Price ($)'
            },
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Top 10 Most Expensive Areas
        top_areas = df.groupby('AreaName')['Price ($)'].mean().nlargest(10).reset_index()
        
        fig = px.bar(
            top_areas,
            x='Price ($)',
            y='AreaName',
            orientation='h',
            title="Top 10 Most Expensive Areas (Average Price)",
            labels={'AreaName': 'Area Name', 'Price ($)': 'Average Price ($)'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


def about_tab():
    """Render the about tab."""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### Ontario House Price Predictor
    
    This application uses machine learning to predict house prices across Ontario based on 
    location data and historical property information.
    
    #### 🎯 Features
    - **Multiple ML Models**: Linear Regression and Random Forest
    - **Interactive Predictions**: Upload data, manual input, or use samples
    - **Real-time Visualization**: Maps, charts, and statistical analysis
    - **Confidence Intervals**: Uncertainty estimates for predictions
    - **Model Comparison**: Evaluate different algorithms
    
    #### 📊 Dataset
    - **Source**: Ontario House Sales Dataset (properties.csv)
    - **Features Used**:
        - Address and Area Name
        - Geographic coordinates (latitude/longitude)
        - Distance from downtown Toronto
        - Area average prices and property counts
    
    #### 🤖 Models
    
    **Linear Regression**
    - Fast and interpretable baseline model
    - Optional Ridge regularization for stability
    - Best for understanding feature relationships
    
    **Random Forest**
    - Ensemble of decision trees
    - Captures non-linear patterns
    - Provides feature importance rankings
    - More accurate but less interpretable
    
    #### 📈 Evaluation Metrics
    - **RMSE**: Measures average prediction error in dollars
    - **MAE**: Median absolute prediction error
    - **R² Score**: Percentage of variance explained (closer to 1.0 is better)
    
    #### 🔧 Preprocessing Steps
    1. Data cleaning (remove invalid prices, handle missing values)
    2. Feature engineering (distance calculations, area statistics)
    3. Categorical encoding (Label Encoding for areas and cities)
    4. Optional feature scaling (StandardScaler)
    
    #### ⚠️ Limitations
    - Predictions based on location data only (no property details like bedrooms, size)
    - Historical data may not reflect current market conditions
    - Accuracy varies by area (better for areas with more data)
    - Outliers and unique properties may not be predicted accurately
    
    #### 🚀 Deployment
    This application is built with **Streamlit** and can be deployed to Streamlit Cloud.
    All models are pre-trained and saved for fast inference.
    
    #### 📚 Technologies Used
    - Python 3.10+
    - Streamlit (UI framework)
    - scikit-learn (ML models)
    - Pandas & NumPy (data processing)
    - Plotly (interactive visualizations)
    - Joblib (model persistence)
    
    #### 👨‍💻 Development Best Practices
    - Modular code structure with separate modules
    - Comprehensive error handling and validation
    - Unit tests for critical functions
    - Type hints and docstrings
    - Caching for performance optimization
    
    ---
    
    **Version**: 1.0.0 | **License**: MIT
    """)
    
    # Help section
    with st.expander("❓ How to Use This App"):
        st.markdown("""
        1. **Configure Model**: Use sidebar to select model and adjust hyperparameters
        2. **Make Predictions**: 
           - Upload CSV with property data, or
           - Enter single property details manually, or
           - Use sample data for quick demo
        3. **View Results**: See predictions with confidence intervals
        4. **Analyze Performance**: Check metrics and compare models
        5. **Explore Visualizations**: View charts and geographic distributions
        6. **Download Results**: Export predictions as CSV
        """)


def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Ontario House Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("*Predict real estate prices using machine learning*")
    
    # Initialize session state
    if 'retrain_models' not in st.session_state:
        st.session_state.retrain_models = False
    
    # Sidebar controls
    model_choice, use_scaling, lr_params, rf_params, random_state = sidebar_controls()
    
    # Load or train models
    if st.session_state.retrain_models:
        all_params = {'use_regularization': False} if model_choice == "Linear Regression" else {'n_estimators': 100, 'max_depth': 20}
        if lr_params:
            all_params.update(lr_params)
        if rf_params:
            all_params.update(rf_params)
        
        train_models_ui(use_scaling, lr_params or {'use_regularization': False}, rf_params or {'n_estimators': 100, 'max_depth': 20}, random_state)
        st.session_state.retrain_models = False
        st.rerun()
    
    # Load models
    model_lr, model_rf, encoders = load_models_and_encoders()
    
    if model_lr is None or model_rf is None:
        st.warning("⚠️ Pre-trained models not found. Training new models...")
        train_models_ui(True, {'use_regularization': False}, {'n_estimators': 100, 'max_depth': 20}, 42)
        st.rerun()
    
    # Select active model
    model = model_lr if model_choice == "Linear Regression" else model_rf
    
    # Load reference data
    df_reference = load_and_prepare_data()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predict",
        "📈 Metrics",
        "📊 Visualizations",
        "ℹ️ About"
    ])
    
    with tab1:
        predict_tab(model, encoders, df_reference)
    
    with tab2:
        metrics_tab(model_choice)
    
    with tab3:
        visualizations_tab(model_choice)
    
    with tab4:
        about_tab()
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Built with Streamlit 🎈 | Machine Learning for Real Estate 🏠 | © 2025
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
