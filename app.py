import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import subprocess
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .main { background-color: #0a0e1a; }
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%); }

    .metric-card {
        background: linear-gradient(135deg, #1a2332 0%, #1e2d40 100%);
        border: 1px solid #2a4060;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,150,255,0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8899aa;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .fraud-alert {
        background: linear-gradient(135deg, #2d0a0a, #3d1010);
        border: 2px solid #ff4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .safe-alert {
        background: linear-gradient(135deg, #0a2d0a, #103d10);
        border: 2px solid #44ff44;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255,68,68,0.3); }
        50% { box-shadow: 0 0 40px rgba(255,68,68,0.6); }
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #00d4ff;
        border-left: 4px solid #00d4ff;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0066cc, #0099ff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0099ff, #00ccff);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,153,255,0.4);
    }
    .stSelectbox > div > div, .stNumberInput > div > div > input, .stSlider {
        background-color: #1a2332 !important;
        border-color: #2a4060 !important;
        color: #e0f0ff !important;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1420 0%, #0a1018 100%);
        border-right: 1px solid #1a2d40;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d1420;
        border-bottom: 2px solid #1a2d40;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8899aa;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
    }
    h1, h2, h3 { color: #e0f0ff !important; }
    p, label { color: #aabbcc !important; }
    .stMarkdown { color: #aabbcc; }
</style>
""", unsafe_allow_html=True)


# ─── Load / Train Models ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    if not os.path.exists('models/random_forest.pkl'):
        with st.spinner("🤖 Training AI models for the first time... (30-60 seconds)"):
            subprocess.run([sys.executable, 'train_model.py'], check=True)

    with open('models/random_forest.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('models/gradient_boosting.pkl', 'rb') as f:
        gb = pickle.load(f)
    with open('models/logistic_regression.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    return rf, gb, lr, scaler, encoders, metrics


def load_dataset():
    """Always reads fresh from disk so new transactions appear immediately"""
    if not os.path.exists('upi_transactions.csv'):
        from generate_dataset import generate_upi_dataset
        df = generate_upi_dataset()
        df.to_csv('upi_transactions.csv', index=False)
    return pd.read_csv('upi_transactions.csv', on_bad_lines='skip')


def save_transaction(upi_id, amount, txn_type, sender_bank, device_type,
                     hour_of_day, txn_freq, location_dist, new_device,
                     failed_attempts, account_age, pin_changed, risk_score):
    """Append a newly checked transaction — always aligned with existing columns"""
    is_fraud = 1 if risk_score >= 50 else 0

    # Read existing CSV to get exact column structure
    if os.path.exists('upi_transactions.csv'):
        existing = pd.read_csv('upi_transactions.csv', on_bad_lines='skip')
        existing_cols = existing.columns.tolist()
    else:
        existing_cols = None

    new_row = {
        'transaction_id': f"TXN{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
        'amount': amount,
        'transaction_type': txn_type,
        'sender_bank': sender_bank,
        'receiver_bank': 'Unknown',
        'device_type': device_type,
        'hour_of_day': hour_of_day,
        'day_of_week': pd.Timestamp.now().weekday(),
        'transaction_freq_1hr': txn_freq,
        'location_distance_km': location_dist,
        'new_device': 1 if new_device == 'Yes' else 0,
        'failed_attempts': failed_attempts,
        'account_age_days': account_age,
        'pin_changed_recently': 1 if pin_changed == 'Yes' else 0,
        'is_weekend': 1 if pd.Timestamp.now().weekday() >= 5 else 0,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'is_fraud': is_fraud,
        'risk_score': risk_score,
        'upi_id': upi_id if upi_id else 'manual_entry',
        'source': 'user_input'
    }

    new_df = pd.DataFrame([new_row])

    if existing_cols is not None:
        # Add any missing columns to existing CSV header first
        for col in new_df.columns:
            if col not in existing_cols:
                existing[col] = ''
                existing_cols.append(col)
        # Align new row to exact same columns
        new_df = new_df.reindex(columns=existing_cols)
        # Save full updated file to fix column alignment permanently
        updated = pd.concat([existing, new_df], ignore_index=True)
        updated.to_csv('upi_transactions.csv', index=False)
    else:
        new_df.to_csv('upi_transactions.csv', index=False)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ UPI Fraud Detection")
    st.markdown("*AI-Powered Transaction Security*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "🎆 Predict Transaction",
        "📊 Model Analytics",
        "📋 Transaction Data"
    ])
    st.markdown("---")
    st.markdown("**Project Info**")
    st.markdown("- 🎓 MCA Final Project")
    st.markdown("- 🏫 GBU,Greater Noida")
    st.markdown("- 🤖 ML Stack: sklearn")
    st.markdown("- 📦 Models: RF, GB, LR")


# ─── Load everything ─────────────────────────────────────────────────────────
rf, gb, lr, scaler, encoders, metrics = load_models()
df = load_dataset()


# ══════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# 🛡️ UPI Fraud Detection System")
    st.markdown("##### Real-time Transaction Monitoring & Fraud Analytics")
    st.markdown("---")

    # KPI Metrics
    total = len(df)
    fraud_count = df['is_fraud'].sum()
    safe_count = total - fraud_count
    fraud_pct = (fraud_count / total) * 100
    total_amount = df['amount'].sum()
    fraud_amount = df[df['is_fraud'] == 1]['amount'].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Transactions</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ff6b6b">{fraud_count:,}</div>
            <div class="metric-label">Fraud Detected</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#44ff88">{safe_count:,}</div>
            <div class="metric-label">Safe Transactions</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ffaa00">{fraud_pct:.1f}%</div>
            <div class="metric-label">Fraud Rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Transaction Distribution</div>', unsafe_allow_html=True)
        fig = px.pie(
            values=[safe_count, fraud_count],
            names=['Legitimate', 'Fraudulent'],
            color_discrete_sequence=['#00d4ff', '#ff4444'],
            hole=0.6
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#aabbcc'),
            legend=dict(font=dict(color='#aabbcc')),
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Fraud by Transaction Type</div>', unsafe_allow_html=True)
        txn_fraud = df.groupby('transaction_type')['is_fraud'].agg(['sum', 'count']).reset_index()
        txn_fraud['fraud_rate'] = (txn_fraud['sum'] / txn_fraud['count'] * 100).round(1)
        fig2 = px.bar(
            txn_fraud, x='transaction_type', y='fraud_rate',
            color='fraud_rate',
            color_continuous_scale=['#00d4ff', '#ff4444'],
            labels={'fraud_rate': 'Fraud Rate (%)', 'transaction_type': 'Type'}
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#aabbcc'), margin=dict(t=10, b=10),
            showlegend=False, coloraxis_showscale=False
        )
        fig2.update_xaxes(gridcolor='#1a2d40')
        fig2.update_yaxes(gridcolor='#1a2d40')
        st.plotly_chart(fig2, use_container_width=True)

    # Charts Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Hourly Fraud Pattern</div>', unsafe_allow_html=True)
        hourly = df.groupby('hour_of_day')['is_fraud'].mean().reset_index()
        fig3 = px.line(
            hourly, x='hour_of_day', y='is_fraud',
            labels={'hour_of_day': 'Hour of Day', 'is_fraud': 'Fraud Rate'},
            line_shape='spline'
        )
        fig3.update_traces(line_color='#ff6b6b', line_width=3,
                           fill='tozeroy', fillcolor='rgba(255,107,107,0.1)')
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#aabbcc'), margin=dict(t=10, b=10)
        )
        fig3.update_xaxes(gridcolor='#1a2d40')
        fig3.update_yaxes(gridcolor='#1a2d40')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Amount vs Fraud Risk</div>', unsafe_allow_html=True)
        sample = df.sample(min(500, len(df)), random_state=42)
        fig4 = px.scatter(
            sample, x='amount', y='location_distance_km',
            color='is_fraud',
            color_discrete_map={0: '#00d4ff', 1: '#ff4444'},
            labels={'amount': 'Amount (₹)', 'location_distance_km': 'Distance (km)', 'is_fraud': 'Fraud'},
            opacity=0.7
        )
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#aabbcc'), margin=dict(t=10, b=10)
        )
        fig4.update_xaxes(gridcolor='#1a2d40')
        fig4.update_yaxes(gridcolor='#1a2d40')
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2: PREDICT TRANSACTION
# ══════════════════════════════════════════════════════════════
elif page == "🎆 Predict Transaction":
    st.markdown("# 🎆 Predict Transaction Fraud")
    st.markdown("##### Choose how you want to analyze — UPI ID lookup or CSV bulk upload")
    st.markdown("---")

    # ── UPI ID Validator ─────────────────────────────────────
    VALID_UPI_HANDLES = [
        'okaxis', 'ybl', 'ibl', 'okhdfcbank', 'okicici', 'oksbi',
        'paytm', 'apl', 'upi', 'axl', 'ikwik', 'freecharge',
        'juspay', 'yapl', 'superyes', 'axisbank', 'hdfcbank',
        'sbi', 'icici', 'pnb', 'kotak', 'indus', 'aubank',
        'idbi', 'boi', 'canara', 'rbl', 'federal', 'jsb',
        'pingpay', 'abfspay', 'timecosmos', 'jupiter', 'slice',
        'niyoicici', 'fifipay', 'rajgovhdfcbank', 'mahb', 'jkb'
    ]

    def validate_upi_id(upi_id):
        import re
        upi_id = upi_id.strip().lower()
        if '@' not in upi_id:
            return False, "❌ Missing '@' symbol", None, None
        parts = upi_id.split('@')
        if len(parts) != 2:
            return False, "❌ Invalid format — multiple '@' found", None, None
        username, handle = parts
        if not username:
            return False, "❌ Username cannot be empty", None, None
        if not re.match(r'^[a-z0-9._-]+$', username):
            return False, "❌ Username contains invalid characters", None, None
        if len(username) < 3:
            return False, "❌ Username too short (min 3 chars)", None, None
        if handle not in VALID_UPI_HANDLES:
            return False, f"⚠️ Unknown bank handle '@{handle}' — may be invalid", None, handle
        # Map handle to bank name
        handle_bank_map = {
            'okaxis': 'Axis Bank', 'ybl': 'Yes Bank', 'ibl': 'IDFC Bank',
            'okhdfcbank': 'HDFC Bank', 'okicici': 'ICICI Bank', 'oksbi': 'SBI',
            'paytm': 'Paytm Payments Bank', 'apl': 'Amazon Pay', 'upi': 'BHIM UPI',
            'axl': 'Axis Bank', 'ikwik': 'Freecharge', 'freecharge': 'Freecharge',
            'axisbank': 'Axis Bank', 'hdfcbank': 'HDFC Bank', 'sbi': 'SBI',
            'icici': 'ICICI Bank', 'pnb': 'PNB', 'kotak': 'Kotak Bank',
            'indus': 'IndusInd Bank', 'aubank': 'AU Small Finance Bank',
            'idbi': 'IDBI Bank', 'boi': 'Bank of India', 'canara': 'Canara Bank',
            'rbl': 'RBL Bank', 'federal': 'Federal Bank', 'jupiter': 'Jupiter (Federal)',
            'slice': 'Slice', 'niyoicici': 'Niyo (ICICI)', 'pingpay': 'Samsung Pay',
        }
        bank = handle_bank_map.get(handle, handle.upper() + ' Bank')
        return True, f"✅ Valid UPI ID — Linked to **{bank}**", bank, handle

    def make_prediction(amount, txn_type, sender_bank, device_type,
                        hour_of_day, txn_freq, location_dist,
                        new_device, failed_attempts, account_age,
                        pin_changed, model_choice):
        try:
            txn_enc = encoders['txn_type'].transform([txn_type])[0]
        except:
            txn_enc = 0
        try:
            bank_enc = encoders['bank'].transform([sender_bank])[0]
        except:
            bank_enc = 0
        try:
            dev_enc = encoders['device'].transform([device_type])[0]
        except:
            dev_enc = 0

        is_weekend = 1 if pd.Timestamp.now().weekday() >= 5 else 0
        day_of_week = pd.Timestamp.now().weekday()

        features = np.array([[
            amount, hour_of_day, day_of_week, txn_freq,
            location_dist,
            1 if new_device == 'Yes' else 0,
            failed_attempts, account_age,
            1 if pin_changed == 'Yes' else 0,
            is_weekend, txn_enc, bank_enc, dev_enc
        ]])

        if model_choice == 'Random Forest':
            prob = rf.predict_proba(features)[0][1]
        elif model_choice == 'Gradient Boosting':
            prob = gb.predict_proba(features)[0][1]
        else:
            features_scaled = scaler.transform(features)
            prob = lr.predict_proba(features_scaled)[0][1]

        return round(prob * 100, 1)

    def show_result(risk_score):
        prediction = 1 if risk_score >= 50 else 0
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            if prediction == 1:
                st.markdown(f"""<div class="fraud-alert">
                    <div style="font-size:3rem">🚨</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#ff4444; margin:8px 0">FRAUDULENT TRANSACTION</div>
                    <div style="font-size:1rem; color:#ff8888">Risk Score: <strong>{risk_score}%</strong></div>
                    <div style="font-size:0.85rem; color:#cc6666; margin-top:8px">This transaction has been flagged as potentially fraudulent. Immediate review recommended.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="safe-alert">
                    <div style="font-size:3rem">✅</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#44ff88; margin:8px 0">LEGITIMATE TRANSACTION</div>
                    <div style="font-size:1rem; color:#88ffaa">Risk Score: <strong>{risk_score}%</strong></div>
                    <div style="font-size:0.85rem; color:#66cc88; margin-top:8px">Transaction appears safe. No suspicious patterns detected.</div>
                </div>""", unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Risk Score", 'font': {'color': '#aabbcc', 'size': 18}},
            number={'suffix': "%", 'font': {'color': '#00d4ff', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#aabbcc'},
                'bar': {'color': '#ff4444' if risk_score > 50 else '#00d4ff'},
                'bgcolor': '#1a2332', 'bordercolor': '#2a4060',
                'steps': [
                    {'range': [0, 30], 'color': '#0a2d0a'},
                    {'range': [30, 60], 'color': '#2d2a0a'},
                    {'range': [60, 100], 'color': '#2d0a0a'}
                ],
                'threshold': {'line': {'color': 'white', 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': '#aabbcc'},
                                height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── MODE SELECTOR ─────────────────────────────────────────
    mode = st.radio("", ["🪪  UPI ID + Manual Entry", "📂  Bulk CSV Upload"],
                    horizontal=True, label_visibility="collapsed")
    st.markdown("")

    # ════════════════════════════════════════════════════════
    # MODE 1: UPI ID + Manual Entry
    # ════════════════════════════════════════════════════════
    if mode == "🪪  UPI ID + Manual Entry":

        # UPI ID input at top
        st.markdown('<div class="section-header">🪪 UPI ID Verification</div>', unsafe_allow_html=True)
        upi_col1, upi_col2 = st.columns([3, 1])
        with upi_col1:
            upi_id = st.text_input("Enter Sender UPI ID", placeholder="e.g. rahul123@okaxis",
                                   help="Format: username@bankhandle")
        with upi_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            verify_btn = st.button("🔍 Verify UPI ID", use_container_width=True)

        upi_verified = False
        detected_bank = 'SBI'

        if upi_id:
            is_valid, message, bank_name, handle = validate_upi_id(upi_id)
            if is_valid:
                st.success(message)
                upi_verified = True
                if bank_name:
                    detected_bank = bank_name
                    # Map bank name to dropdown option
                    bank_map = {
                        'SBI': 'SBI', 'HDFC Bank': 'HDFC', 'ICICI Bank': 'ICICI',
                        'Axis Bank': 'Axis', 'Kotak Bank': 'Kotak', 'PNB': 'PNB',
                        'Bank of India': 'BOI', 'Canara Bank': 'Canara',
                        'Yes Bank': 'HDFC', 'IDFC Bank': 'Axis',
                        'Paytm Payments Bank': 'SBI', 'Amazon Pay': 'Axis'
                    }
                    detected_bank = bank_map.get(bank_name, 'SBI')
                    st.info(f"🏦 Bank auto-detected: **{bank_name}** — pre-filled below")
            else:
                st.error(message)

        st.markdown("---")
        st.markdown('<div class="section-header">📋 Transaction Details</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            amount = st.number_input("💰 Transaction Amount (₹)", min_value=1.0,
                                     max_value=500000.0, value=5000.0, step=100.0)
            txn_type = st.selectbox("📱 Transaction Type",
                                    ['P2P', 'P2M', 'Bill Payment', 'Recharge', 'Shopping'])
            bank_options = ['SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak', 'PNB', 'BOI', 'Canara']
            default_bank_idx = bank_options.index(detected_bank) if detected_bank in bank_options else 0
            sender_bank = st.selectbox("🏦 Sender Bank", bank_options, index=default_bank_idx,
                                       help="Auto-filled from UPI ID if verified")
            device_type = st.selectbox("📲 Device Type", ['Android', 'iOS', 'Web'])
            hour_of_day = st.slider("🕐 Hour of Transaction", 0, 23,
                                    pd.Timestamp.now().hour)
            model_choice = st.selectbox("🤖 Select ML Model",
                                        ['Random Forest', 'Gradient Boosting', 'Logistic Regression'])

        with col_right:
            st.markdown('<div class="section-header">⚠️ Risk Indicators</div>', unsafe_allow_html=True)
            txn_freq    = st.slider("⚡ Transactions in Last Hour", 0, 20, 1)
            location_dist = st.slider("📍 Distance Sender-Receiver (km)", 0, 3000, 50)
            new_device  = st.radio("📱 New/Unknown Device?", ['No', 'Yes'])
            failed_attempts = st.slider("❌ Failed Login Attempts", 0, 5, 0)
            account_age = st.number_input("📅 Account Age (days)", min_value=1,
                                          max_value=3650, value=365)
            pin_changed = st.radio("🔑 PIN Changed Recently?", ['No', 'Yes'])

        st.markdown("")
        btn_col = st.columns([1, 2, 1])[1]
        with btn_col:
            predict_btn = st.button("🔍 ANALYZE TRANSACTION", use_container_width=True)

        if predict_btn:
            if upi_id and not upi_verified:
                st.warning("⚠️ UPI ID entered but not verified. Please fix the UPI ID or clear it.")
            else:
                risk_score = make_prediction(
                    amount, txn_type, sender_bank, device_type,
                    hour_of_day, txn_freq, location_dist,
                    new_device, failed_attempts, account_age,
                    pin_changed, model_choice
                )
                st.markdown("")
                if upi_id and upi_verified:
                    st.markdown(f"**Analyzing UPI ID:** `{upi_id.strip().lower()}`")
                show_result(risk_score)
                # ── Save to dataset ──────────────────────────
                save_transaction(
                    upi_id, amount, txn_type, sender_bank, device_type,
                    hour_of_day, txn_freq, location_dist, new_device,
                    failed_attempts, account_age, pin_changed, risk_score
                )
                st.success("💾 Transaction saved to dataset — visible in Transaction Data page")

    # ════════════════════════════════════════════════════════
    # MODE 2: CSV BULK UPLOAD
    # ════════════════════════════════════════════════════════
    else:
        st.markdown('<div class="section-header">📂 Bulk CSV Transaction Analysis</div>',
                    unsafe_allow_html=True)

        # Expected format info
        with st.expander("📌 Expected CSV Format — Click to expand"):
            st.markdown("""
Your CSV should have these columns:

| Column | Example |
|--------|---------|
| `amount` | 5000.0 |
| `transaction_type` | P2P / P2M / Bill Payment / Recharge / Shopping |
| `sender_bank` | SBI / HDFC / ICICI / Axis / Kotak / PNB / BOI / Canara |
| `device_type` | Android / iOS / Web |
| `hour_of_day` | 0–23 |
| `transaction_freq_1hr` | 0–20 |
| `location_distance_km` | 0–3000 |
| `new_device` | 0 or 1 |
| `failed_attempts` | 0–5 |
| `account_age_days` | 1–3650 |
| `pin_changed_recently` | 0 or 1 |

> 💡 You can also download and use our **sample dataset** from the Transaction Data page.
            """)

        model_choice_csv = st.selectbox("🤖 Select ML Model",
                                        ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
                                        key='csv_model')

        uploaded_file = st.file_uploader("Upload your transaction CSV file",
                                         type=['csv'],
                                         help="Max file size: 200MB")

        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded: **{len(upload_df)} transactions** found")

                required_cols = ['amount', 'hour_of_day', 'transaction_freq_1hr',
                                 'location_distance_km', 'new_device', 'failed_attempts',
                                 'account_age_days', 'pin_changed_recently']

                missing = [c for c in required_cols if c not in upload_df.columns]
                if missing:
                    st.error(f"❌ Missing columns: {', '.join(missing)}")
                else:
                    if st.button("🚀 ANALYZE ALL TRANSACTIONS", use_container_width=True):
                        with st.spinner("🤖 Analyzing all transactions..."):
                            results = []
                            for _, row in upload_df.iterrows():
                                txn_type_val = row.get('transaction_type', 'P2P')
                                bank_val     = row.get('sender_bank', 'SBI')
                                device_val   = row.get('device_type', 'Android')

                                try:
                                    txn_enc = encoders['txn_type'].transform([txn_type_val])[0]
                                except: txn_enc = 0
                                try:
                                    bank_enc = encoders['bank'].transform([bank_val])[0]
                                except: bank_enc = 0
                                try:
                                    dev_enc = encoders['device'].transform([device_val])[0]
                                except: dev_enc = 0

                                is_weekend = row.get('is_weekend', 0)
                                day_of_week = row.get('day_of_week', 0)

                                feat = np.array([[
                                    row['amount'], row['hour_of_day'], day_of_week,
                                    row['transaction_freq_1hr'], row['location_distance_km'],
                                    row['new_device'], row['failed_attempts'],
                                    row['account_age_days'], row['pin_changed_recently'],
                                    is_weekend, txn_enc, bank_enc, dev_enc
                                ]])

                                if model_choice_csv == 'Random Forest':
                                    prob = rf.predict_proba(feat)[0][1]
                                elif model_choice_csv == 'Gradient Boosting':
                                    prob = gb.predict_proba(feat)[0][1]
                                else:
                                    feat_s = scaler.transform(feat)
                                    prob = lr.predict_proba(feat_s)[0][1]

                                results.append(round(prob * 100, 1))

                            upload_df['risk_score'] = results
                            upload_df['status'] = upload_df['risk_score'].apply(
                                lambda x: '🚨 Fraud' if x >= 50 else '✅ Legitimate'
                            )

                        # Summary KPIs
                        total_txns  = len(upload_df)
                        fraud_txns  = (upload_df['risk_score'] >= 50).sum()
                        safe_txns   = total_txns - fraud_txns
                        avg_risk    = upload_df['risk_score'].mean()

                        k1, k2, k3, k4 = st.columns(4)
                        with k1:
                            st.markdown(f"""<div class="metric-card">
                                <div class="metric-value">{total_txns}</div>
                                <div class="metric-label">Total Analyzed</div>
                            </div>""", unsafe_allow_html=True)
                        with k2:
                            st.markdown(f"""<div class="metric-card">
                                <div class="metric-value" style="color:#ff6b6b">{fraud_txns}</div>
                                <div class="metric-label">Fraud Detected</div>
                            </div>""", unsafe_allow_html=True)
                        with k3:
                            st.markdown(f"""<div class="metric-card">
                                <div class="metric-value" style="color:#44ff88">{safe_txns}</div>
                                <div class="metric-label">Safe</div>
                            </div>""", unsafe_allow_html=True)
                        with k4:
                            st.markdown(f"""<div class="metric-card">
                                <div class="metric-value" style="color:#ffaa00">{avg_risk:.1f}%</div>
                                <div class="metric-label">Avg Risk Score</div>
                            </div>""", unsafe_allow_html=True)

                        st.markdown("")

                        # Risk distribution chart
                        fig_hist = px.histogram(
                            upload_df, x='risk_score', nbins=20,
                            color_discrete_sequence=['#00d4ff'],
                            labels={'risk_score': 'Risk Score (%)', 'count': 'Transactions'},
                            title='Risk Score Distribution'
                        )
                        fig_hist.add_vline(x=50, line_dash="dash", line_color="#ff4444",
                                           annotation_text="Fraud Threshold")
                        fig_hist.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#aabbcc'), margin=dict(t=40, b=10)
                        )
                        fig_hist.update_xaxes(gridcolor='#1a2d40')
                        fig_hist.update_yaxes(gridcolor='#1a2d40')
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Results table
                        st.markdown('<div class="section-header">📋 Detailed Results</div>',
                                    unsafe_allow_html=True)
                        display_cols = ['amount', 'transaction_type', 'sender_bank',
                                        'risk_score', 'status'] if 'transaction_type' in upload_df.columns \
                                       else ['amount', 'risk_score', 'status']
                        st.dataframe(upload_df[display_cols].head(500),
                                     use_container_width=True, hide_index=True)

                        # Download results
                        csv_out = upload_df.to_csv(index=False)
                        st.download_button("⬇️ Download Results CSV", csv_out,
                                           "fraud_analysis_results.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        else:
            st.markdown("""
            <div style="border: 2px dashed #2a4060; border-radius:16px; padding:40px; text-align:center; color:#8899aa;">
                <div style="font-size:3rem">📂</div>
                <div style="font-size:1.1rem; margin-top:8px">Drag & drop your CSV file here</div>
                <div style="font-size:0.85rem; margin-top:4px">or click Browse to select</div>
                <div style="font-size:0.8rem; margin-top:16px; color:#556677;">
                    Tip: Download sample data from the <strong>Transaction Data</strong> page to test this feature
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3: MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Analytics":
    st.markdown("# 📊 Model Performance Analytics")
    st.markdown("---")

    # Accuracy comparison
    model_names = ['Random Forest', 'Gradient Boosting', 'Logistic Regression']
    accuracies = [
        metrics['random_forest']['accuracy'],
        metrics['gradient_boosting']['accuracy'],
        metrics['logistic_regression']['accuracy']
    ]
    aucs = [
        metrics['random_forest']['auc'],
        metrics['gradient_boosting']['auc'],
        metrics['logistic_regression']['auc']
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Model Accuracy Comparison</div>', unsafe_allow_html=True)
        fig = px.bar(
            x=model_names, y=accuracies,
            color=accuracies,
            color_continuous_scale=['#0066cc', '#00d4ff'],
            labels={'x': 'Model', 'y': 'Accuracy (%)'},
            text=[f"{a}%" for a in accuracies]
        )
        fig.update_traces(textposition='outside', textfont=dict(color='white'))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#aabbcc'), margin=dict(t=10, b=10),
            yaxis_range=[80, 100], coloraxis_showscale=False
        )
        fig.update_xaxes(gridcolor='#1a2d40')
        fig.update_yaxes(gridcolor='#1a2d40')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">AUC-ROC Scores</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        colors = ['#00d4ff', '#ff6b6b', '#44ff88']
        for i, (name, auc) in enumerate(zip(model_names, aucs)):
            fig2.add_trace(go.Bar(
                name=name, x=[name], y=[auc],
                marker_color=colors[i],
                text=[f"{auc:.4f}"], textposition='outside'
            ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#aabbcc'), margin=dict(t=10, b=10),
            yaxis_range=[0.8, 1.0], showlegend=False
        )
        fig2.update_xaxes(gridcolor='#1a2d40')
        fig2.update_yaxes(gridcolor='#1a2d40')
        st.plotly_chart(fig2, use_container_width=True)

    # Feature Importance
    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    feat_imp = metrics['feature_importance']
    feat_df = pd.DataFrame(list(feat_imp.items()), columns=['Feature', 'Importance'])
    feat_df = feat_df.sort_values('Importance', ascending=True)

    fig3 = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale=['#0066cc', '#00d4ff', '#ff6b6b'])
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#aabbcc'), margin=dict(t=10, b=10),
        height=400, coloraxis_showscale=False
    )
    fig3.update_xaxes(gridcolor='#1a2d40')
    fig3.update_yaxes(gridcolor='#1a2d40')
    st.plotly_chart(fig3, use_container_width=True)

    # Metrics Table
    st.markdown('<div class="section-header">Detailed Metrics</div>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy (%)': accuracies,
        'AUC-ROC': aucs,
        'Status': ['✅ Best' if a == max(accuracies) else '🔵 Good' for a in accuracies]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4: TRANSACTION DATA
# ══════════════════════════════════════════════════════════════
elif page == "📋 Transaction Data":
    st.markdown("# 📋 Transaction Dataset")
    st.markdown("---")

    # Always load fresh from disk to reflect new transactions
    df_live = load_dataset()

    # Separate user-submitted vs generated
    user_txns = df_live[df_live.get('source', pd.Series([''] * len(df_live))) == 'user_input'] \
        if 'source' in df_live.columns else pd.DataFrame()
    total_user = len(user_txns)

    # KPI row
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df_live):,}</div>
            <div class="metric-label">Total Records</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#00d4ff">{total_user}</div>
            <div class="metric-label">User Submitted</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        fraud_live = df_live['is_fraud'].sum()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ff6b6b">{fraud_live:,}</div>
            <div class="metric-label">Fraud Records</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_fraud = st.selectbox("Filter by Status", ['All', 'Fraudulent Only', 'Legitimate Only'])
    with col2:
        filter_type = st.selectbox("Filter by Type", ['All'] + sorted(df_live['transaction_type'].dropna().unique().tolist()))
    with col3:
        filter_bank = st.selectbox("Filter by Bank", ['All'] + sorted(df_live['sender_bank'].dropna().unique().tolist()))
    with col4:
        filter_source = st.selectbox("Filter by Source",
                                     ['All', 'User Submitted', 'Generated Data']
                                     if 'source' in df_live.columns else ['All'])

    # Apply filters
    filtered = df_live.copy()
    # Put user-submitted rows on top
    if 'source' in filtered.columns:
        filtered = pd.concat([
            filtered[filtered['source'] == 'user_input'],
            filtered[filtered['source'] != 'user_input']
        ]).reset_index(drop=True)

    if filter_fraud == 'Fraudulent Only':
        filtered = filtered[filtered['is_fraud'] == 1]
    elif filter_fraud == 'Legitimate Only':
        filtered = filtered[filtered['is_fraud'] == 0]
    if filter_type != 'All':
        filtered = filtered[filtered['transaction_type'] == filter_type]
    if filter_bank != 'All':
        filtered = filtered[filtered['sender_bank'] == filter_bank]
    if 'source' in filtered.columns and filter_source == 'User Submitted':
        filtered = filtered[filtered['source'] == 'user_input']
    elif 'source' in filtered.columns and filter_source == 'Generated Data':
        filtered = filtered[filtered['source'] != 'user_input']

    st.markdown(f"**Showing {len(filtered):,} of {len(df_live):,} transactions** "
                f"{'— 🆕 User submitted entries shown at top' if total_user > 0 else ''}")

    # Build display columns dynamically
    base_cols = ['transaction_id', 'amount', 'transaction_type', 'sender_bank',
                 'device_type', 'hour_of_day', 'is_fraud', 'timestamp']
    if 'upi_id' in filtered.columns:
        base_cols.insert(1, 'upi_id')
    if 'risk_score' in filtered.columns:
        base_cols.append('risk_score')

    available_cols = [c for c in base_cols if c in filtered.columns]
    display_df = filtered[available_cols].head(300).copy()
    display_df['is_fraud'] = display_df['is_fraud'].map({0: '✅ Legitimate', 1: '🚨 Fraud'})
    display_df['amount'] = display_df['amount'].apply(lambda x: f"₹{x:,.2f}")
    if 'risk_score' in display_df.columns:
        display_df['risk_score'] = display_df['risk_score'].apply(
            lambda x: f"{x}%" if pd.notna(x) else 'N/A'
        )

    col_rename = {
        'transaction_id': 'Txn ID', 'upi_id': 'UPI ID', 'amount': 'Amount',
        'transaction_type': 'Type', 'sender_bank': 'Bank', 'device_type': 'Device',
        'hour_of_day': 'Hour', 'is_fraud': 'Status', 'timestamp': 'Timestamp',
        'risk_score': 'Risk Score'
    }
    display_df.rename(columns=col_rename, inplace=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

    dl1, dl2 = st.columns(2)
    with dl1:
        csv = filtered.to_csv(index=False)
        st.download_button("⬇️ Download Filtered CSV", csv, "upi_transactions.csv", "text/csv")
    with dl2:
        if total_user > 0 and 'source' in df_live.columns:
            user_csv = df_live[df_live['source'] == 'user_input'].to_csv(index=False)
            st.download_button("⬇️ Download User Submissions Only", user_csv,
                               "user_submitted_transactions.csv", "text/csv")
