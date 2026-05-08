import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from google import genai
from google.genai import types

# Load environment variables (locally from .env, on cloud from Secrets)
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="AutoValue - Expert Car Appraisal", page_icon="🚘", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 30px;
    }
    
    .price-box {
        background: linear-gradient(135deg, #1e1e2d 0%, #151521 100%);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        border: 1px solid #323248;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    
    .price-number {
        font-size: 4rem;
        font-weight: 800;
        color: #48bb78;
        margin: 0;
    }
    
    .verdict-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    .verdict-great { background-color: rgba(72, 187, 120, 0.2); color: #48bb78; border: 1px solid #48bb78; }
    .verdict-fair { background-color: rgba(237, 137, 54, 0.2); color: #ed8936; border: 1px solid #ed8936; }
    .verdict-overpriced { background-color: rgba(229, 62, 62, 0.2); color: #e53e3e; border: 1px solid #e53e3e; }
    
    .expert-take {
        background: rgba(45, 55, 72, 0.4);
        border-left: 4px solid #805ad5;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #E2E8F0;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .comp-card {
        background-color: #2D3748;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #4FD1C5;
    }
    
    .comp-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: white;
        margin-bottom: 5px;
    }
    
    .comp-details {
        font-size: 0.9rem;
        color: #A0AEC0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.joblib")
        dataset = joblib.load("dataset.joblib")
    except Exception as e:
        st.warning("Cloud server environment mismatch detected. Rebuilding machine learning model dynamically... (This only happens once!)")
        import train_model
        train_model.main()
        model = joblib.load("model.joblib")
        dataset = joblib.load("dataset.joblib")
    
    # Fit NearestNeighbors for comparable cars
    nn_features = ['year', 'miles_driven', 'engine']
    X_nn = dataset[nn_features].dropna()
    scaler = StandardScaler()
    X_nn_scaled = scaler.fit_transform(X_nn)
    
    nn_model = NearestNeighbors(n_neighbors=3, metric='euclidean')
    nn_model.fit(X_nn_scaled)
    
    return model, dataset, nn_model, scaler, X_nn.index

model, dataset, nn_model, nn_scaler, nn_indices = load_assets()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input("Gemini API Key", type="password", value=api_key if api_key and api_key != "your_api_key_here" else "", help="Overrides .env. Get yours at aistudio.google.com")
    if user_api_key:
        api_key = user_api_key

# --- UI Header ---
st.markdown('<p class="hero-title">AutoValue</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Expert AI-Powered Vehicle Appraisal Engine</p>', unsafe_allow_html=True)

# --- Main Layout ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Vehicle Details")
    
    make_options = sorted(dataset['make'].unique())
    make = st.selectbox("Make", make_options, index=make_options.index("Maruti") if "Maruti" in make_options else 0)
    model_options = sorted(dataset[dataset['make'] == make]['model'].unique())
    car_model = st.selectbox("Model", model_options)
    
    with st.form("appraisal_form"):
        # Form inputs
        c1, c2 = st.columns(2)
        year = c1.number_input("Year", min_value=1990, max_value=2024, value=2018, step=1)
        miles_driven = c2.number_input("Miles Driven", min_value=0.0, value=45000.0, step=1000.0)
        
        c3, c4 = st.columns(2)
        fuel = c3.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        transmission = c4.selectbox("Transmission", ["Manual", "Automatic"])
        
        c5, c6 = st.columns(2)
        seller_type = c5.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
        owner = c6.selectbox("Owner History", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        
        c7, c8 = st.columns(2)
        engine = c7.number_input("Engine (CC)", min_value=500.0, value=1500.0, step=100.0)
        max_power = c8.number_input("Max Power (bhp)", min_value=30.0, value=100.0, step=10.0)
        
        c9, c10 = st.columns(2)
        mileage_mpg = c9.number_input("Mileage (MPG)", min_value=10.0, value=40.0, step=1.0)
        seats = c10.number_input("Seats", min_value=2.0, max_value=10.0, value=5.0, step=1.0)
        
        st.markdown("---")
        asking_price = st.number_input("Asking Price (Optional, USD)", min_value=0.0, value=0.0, step=500.0)
        
        submit_button = st.form_submit_button("Appraise Vehicle", use_container_width=True)

with col2:
    if submit_button:
        # Prepare input dataframe
        input_data = pd.DataFrame({
            'make': [make],
            'year': [year],
            'miles_driven': [miles_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'engine': [engine],
            'max_power': [max_power],
            'mileage_mpg': [mileage_mpg],
            'seats': [seats]
        })
        
        with st.spinner("Analyzing market data..."):
            # 1. Prediction
            prediction = model.predict(input_data)[0]
            
            # 2. Verdict
            verdict_html = ""
            if asking_price > 0:
                if asking_price > prediction * 1.10:
                    verdict_html = '<div class="verdict-badge verdict-overpriced">⚠️ Overpriced</div>'
                elif asking_price < prediction * 0.90:
                    verdict_html = '<div class="verdict-badge verdict-great">🔥 Great Buy</div>'
                else:
                    verdict_html = '<div class="verdict-badge verdict-fair">✅ Fair Deal</div>'
            
            # Show Price & Verdict
            st.markdown(f"""
            <div class="price-box">
                <p style="color: #A0AEC0; margin-bottom: 5px; font-size: 1.1rem;">Estimated Fair Market Value</p>
                <h1 class="price-number">${prediction:,.0f}</h1>
                {verdict_html}
            </div>
            """, unsafe_allow_html=True)
            
            # 3. Gemini Expert Take
            st.subheader("Expert Analysis")
            if not api_key or api_key == "your_api_key_here":
                st.warning("Gemini API key is not configured. Please add it to the .env file to view the expert analysis.")
            else:
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
                    You are an expert used car appraiser with 20 years of experience in the Indian automotive market. 
                    You have just received a machine learning model's price prediction for a vehicle. 
                    Your job is not to repeat the number — your job is to explain the story behind it. 
                    What makes this car worth what it's worth? What should the buyer watch out for? Is this a smart purchase? 
                    Speak like a trusted advisor, not a calculator. Be specific, be direct, and be honest.
                    
                    Car Details:
                    - Make & Model: {make} {car_model}
                    - Year: {year}
                    - Miles Driven: {miles_driven:,.0f}
                    - Fuel: {fuel}
                    - Transmission: {transmission}
                    - Engine: {engine} CC, {max_power} bhp
                    - Owner History: {owner}
                    - Asking Price: {"$" + str(asking_price) if asking_price > 0 else "Not specified"}
                    - Model Predicted Fair Value: ${prediction:,.0f}
                    
                    Provide a 3-4 sentence expert take on this valuation.
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt
                    )
                    
                    st.markdown(f'<div class="expert-take">"{response.text.strip()}"</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating expert analysis: {str(e)}")
            
            # 4. Comparable Cars
            st.subheader("Top 3 Comparable Cars")
            
            # Scale input for NN
            input_nn = pd.DataFrame({'year': [year], 'miles_driven': [miles_driven], 'engine': [engine]})
            input_nn_scaled = nn_scaler.transform(input_nn)
            
            distances, indices = nn_model.kneighbors(input_nn_scaled)
            
            # Get the actual rows from the dataset
            comp_indices = nn_indices[indices[0]]
            comparables = dataset.loc[comp_indices]
            
            for _, row in comparables.iterrows():
                st.markdown(f"""
                <div class="comp-card">
                    <div class="comp-title">{row['name']}</div>
                    <div class="comp-details">
                        {row['year']} • {row['miles_driven']:,.0f} miles • {row['engine']:.0f} CC • {row['fuel']} • {row['transmission']} 
                        <span style="color: #48bb78; font-weight: bold; float: right;">${row['selling_price_usd']:,.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        st.info("Enter vehicle details and click 'Appraise Vehicle' to see the valuation.")
