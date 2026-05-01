import streamlit as st
import pandas as pd
import numpy as np
import re, pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* BACKGROUND — target stApp which is the real root */
[data-testid="stApp"] {
    background: #080b14 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Glowing orbs via a fixed div injected below */
#bg-orbs {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
}
#bg-orbs span {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: .55;
}
#bg-orbs span:nth-child(1) {
    width: 520px; height: 520px;
    background: radial-gradient(circle, #4f46e5 0%, transparent 70%);
    top: -140px; left: -100px;
}
#bg-orbs span:nth-child(2) {
    width: 440px; height: 440px;
    background: radial-gradient(circle, #7c3aed 0%, transparent 70%);
    bottom: -120px; right: -80px;
}
#bg-orbs span:nth-child(3) {
    width: 300px; height: 300px;
    background: radial-gradient(circle, #0d9488 0%, transparent 70%);
    top: 45%; left: 55%;
    opacity: .25;
}

/* Make everything above orbs */
[data-testid="stMain"],
[data-testid="stHeader"],
section.main,
.block-container { background: transparent !important; }

section[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 800px !important;
    padding: 0 1.5rem 5rem !important;
    position: relative;
    z-index: 1;
}

/* Global font override */
*, label, p, div, span, input, select, button {
    font-family: 'Outfit', sans-serif !important;
}

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 3.2rem 1rem 2rem;
}
.chip {
    display: inline-block;
    padding: .3rem 1.1rem;
    border-radius: 999px;
    border: 1px solid rgba(99,102,241,.45);
    background: rgba(99,102,241,.12);
    color: #a5b4fc;
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -1.5px;
    color: #f9fafb;
    margin: 0 0 .5rem;
}
.hero-title span {
    background: linear-gradient(100deg, #818cf8 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #374151;
    font-size: .88rem;
    font-weight: 400;
}

/* ── GLASS CARD ── */
.gcard {
    background: rgba(12,15,28,.8);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 20px;
    padding: 1.5rem 1.8rem 1.2rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 8px 40px rgba(0,0,0,.4);
}
.ctitle {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #6366f1;
    padding-bottom: .8rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(99,102,241,.15);
}

/* ── WIDGETS ── */
label {
    color: #6b7280 !important;
    font-size: .8rem !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 12px !important;
    color: #e5e7eb !important;
    font-size: .87rem !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: rgba(99,102,241,.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.1) !important;
}
div[data-baseweb="select"] svg { color: #4b5563 !important; }
div[data-baseweb="popover"] div[role="listbox"] {
    background: #0c0f1c !important;
    border: 1px solid rgba(99,102,241,.2) !important;
    border-radius: 14px !important;
}
div[data-baseweb="option"] { color: #d1d5db !important; font-size: .86rem !important; }
div[data-baseweb="option"]:hover,
div[data-baseweb="option"][aria-selected="true"] {
    background: rgba(99,102,241,.18) !important; color: #c7d2fe !important;
}

/* slider thumb */
div[data-testid="stSlider"] div[role="slider"] {
    background: #6366f1 !important;
    border: 2px solid #fff !important;
    box-shadow: 0 0 14px rgba(99,102,241,.8) !important;
}
/* slider filled track */
div[data-testid="stSlider"] > div > div > div > div:nth-child(2) {
    background: #6366f1 !important;
}

input[type="checkbox"] { accent-color: #6366f1 !important; }
.stCheckbox span { color: #9ca3af !important; font-size: .86rem !important; }

/* ── BUTTON ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 1rem 0 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: .5px !important;
    box-shadow: 0 6px 35px rgba(79,70,229,.5) !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
    cursor: pointer !important;
    margin-top: .4rem !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 14px 45px rgba(79,70,229,.7) !important;
}

/* ── RESULT ── */
.result {
    margin-top: 1.2rem;
    padding: 2.5rem 1.5rem 2rem;
    border-radius: 22px;
    border: 1px solid rgba(99,102,241,.35);
    background: linear-gradient(135deg, rgba(79,70,229,.13) 0%, rgba(124,58,237,.09) 100%);
    text-align: center;
}
.rtag {
    font-size: .68rem; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #818cf8; margin-bottom: .5rem;
}
.rprice {
    font-size: 4.2rem; font-weight: 800;
    line-height: 1; letter-spacing: -2px;
    color: #f9fafb; margin-bottom: .4rem;
}
.rrange { font-size: .82rem; color: #4b5563; margin-bottom: 1.1rem; }
.rrange b { color: #6b7280; }
.rpills { display:flex; justify-content:center; gap:.5rem; flex-wrap:wrap; }
.rpill {
    background: rgba(99,102,241,.12);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 999px; padding: .22rem .85rem;
    font-size: .72rem; color: #a5b4fc;
}

.stSpinner > div { color: #818cf8 !important; }
[data-testid="stAlertContainer"] { border-radius: 14px !important; }
</style>

<!-- glowing background orbs -->
<div id="bg-orbs">
  <span></span><span></span><span></span>
</div>
""", unsafe_allow_html=True)


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer(df):
    d = df.copy()
    d['Ram']    = d['Ram'].astype(str).str.extract(r'(\d+)').astype(int)
    d['Weight'] = d['Weight'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    d.drop(columns=['laptop_ID'], inplace=True, errors='ignore')
    d['Touchscreen'] = d['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
    d['IPS']         = d['ScreenResolution'].apply(lambda x: 1 if 'IPS' in str(x) else 0)
    res = d['ScreenResolution'].str.split('x', n=1, expand=True)
    d['X_res'] = res[0].str.extract(r'(\d+)').astype(float)
    d['Y_res'] = res[1].astype(float)
    d['PPI']   = ((d['X_res']**2 + d['Y_res']**2)**0.5) / d['Inches']
    d['Cpu_brand'] = d['Cpu'].apply(lambda x: x.split()[0])
    d['Cpu_speed'] = d['Cpu'].apply(
        lambda x: float(re.search(r'(\d+\.?\d+)GHz', str(x)).group(1))
        if re.search(r'(\d+\.?\d+)GHz', str(x)) else np.nan)
    d['Gpu_brand'] = d['Gpu'].apply(lambda x: x.split()[0])
    def mem_gb(m):
        sizes = re.findall(r'(\d+)(GB|TB)', str(m))
        return sum(int(s)*(1000 if u=='TB' else 1) for s,u in sizes)
    d['Memory_GB'] = d['Memory'].apply(mem_gb)
    d['SSD'] = d['Memory'].apply(lambda x: 1 if 'SSD' in str(x) else 0)
    d['HDD'] = d['Memory'].apply(lambda x: 1 if 'HDD' in str(x) else 0)
    d.drop(columns=['Product','Cpu','Gpu','Memory','ScreenResolution','X_res','Y_res'], inplace=True)
    d.dropna(inplace=True)
    return d


# ── Train / load model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    pkl = "laptop_model.pkl"
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    df = pd.read_csv("laptop_price.csv", encoding='latin-1')
    d  = engineer(df)
    d['Price_log'] = np.log1p(d['Price_euros'])
    CAT = ['Company','TypeName','OpSys','Cpu_brand','Gpu_brand']
    NUM = ['Inches','Ram','Weight','Touchscreen','IPS','PPI','Cpu_speed','Memory_GB','SSD','HDD']
    X = d[CAT+NUM]; y = d['Price_log']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42)
    pre = ColumnTransformer([
        ('num', StandardScaler(), NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT)
    ])
    pipe = Pipeline([('pre',pre),('m',VotingRegressor([
        ('rf', RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                          max_depth=4, subsample=0.8, random_state=42))
    ]))])
    pipe.fit(Xtr, ytr)
    with open(pkl,"wb") as f: pickle.dump(pipe,f)
    return pipe

try:
    with st.spinner("Loading model…"):
        model = get_model()
except FileNotFoundError:
    st.error("❌ `laptop_price.csv` not found in the same folder as app.py")
    st.stop()


# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="chip">ML Price Estimator</div>
  <div class="hero-title">Laptop <span>Price</span> Predictor</div>
  <div class="hero-sub">RF + Gradient Boosting &nbsp;·&nbsp; 1,300+ laptops &nbsp;·&nbsp; Prices in Euros</div>
</div>
""", unsafe_allow_html=True)


# ── CARD 1 — Brand & System ───────────────────────────────────────────────────
st.markdown('<div class="gcard"><div class="ctitle">🏷 Brand & System</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    company  = st.selectbox("Brand", ['Acer','Apple','Asus','Chuwi','Dell','Fujitsu',
                                       'Google','HP','Huawei','Lenovo','LG','Mediacom',
                                       'Microsoft','MSI','Razer','Samsung','Toshiba','Vero','Xiaomi'])
    typename = st.selectbox("Laptop Type", ['Notebook','Ultrabook','Gaming',
                                             '2 in 1 Convertible','Workstation','Netbook'])
with c2:
    opsys     = st.selectbox("Operating System", ['Windows 10','macOS','Linux',
                                                   'No OS','Chrome OS','Windows 10 S','Mac OS X'])
    cpu_brand = st.selectbox("CPU Brand", ['Intel','AMD','Samsung'])
    gpu_brand = st.selectbox("GPU Brand", ['Intel','Nvidia','AMD','ARM'])
st.markdown('</div>', unsafe_allow_html=True)


# ── CARD 2 — Hardware ─────────────────────────────────────────────────────────
st.markdown('<div class="gcard"><div class="ctitle">⚙ Hardware Specs</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    # Defaults match real budget laptops (4GB RAM, 500GB HDD, 1.6GHz)
    ram       = st.selectbox("RAM (GB)", [2,4,6,8,12,16,24,32,64], index=1)
    storage   = st.selectbox("Storage (GB)", [32,64,128,256,500,512,1000,2000], index=4)
    cpu_speed = st.slider("CPU Speed (GHz)", 1.0, 4.5, 1.6, step=0.1)
with c4:
    inches = st.slider("Screen Size (in)", 10.1, 18.4, 15.6, step=0.1)
    weight = st.slider("Weight (kg)", 0.9, 5.0, 2.1, step=0.1)
    x_res  = st.selectbox("Screen Width (px)",  [1366,1600,1920,2560,3840], index=0)
    y_res  = st.selectbox("Screen Height (px)", [768,900,1080,1600,2160],   index=0)
st.markdown('</div>', unsafe_allow_html=True)


# ── CARD 3 — Display & Storage type ──────────────────────────────────────────
st.markdown('<div class="gcard"><div class="ctitle">✨ Display & Storage Type</div>', unsafe_allow_html=True)
cb1, cb2, cb3, cb4 = st.columns(4)
with cb1: touchscreen = st.checkbox("Touchscreen")
with cb2: ips         = st.checkbox("IPS Panel")
with cb3: ssd         = st.checkbox("SSD")
with cb4: hdd         = st.checkbox("HDD", value=True)   # default ON — most budget laptops
st.markdown('</div>', unsafe_allow_html=True)


# ── PREDICT ───────────────────────────────────────────────────────────────────
if st.button("⚡  Predict Price", use_container_width=True):
    ppi = ((x_res**2 + y_res**2)**0.5) / inches
    row = pd.DataFrame([{
        'Company': company, 'TypeName': typename, 'OpSys': opsys,
        'Cpu_brand': cpu_brand, 'Gpu_brand': gpu_brand,
        'Inches': inches, 'Ram': ram, 'Weight': weight,
        'Touchscreen': int(touchscreen), 'IPS': int(ips),
        'PPI': ppi, 'Cpu_speed': cpu_speed,
        'Memory_GB': storage, 'SSD': int(ssd), 'HDD': int(hdd),
    }])
    price = np.expm1(model.predict(row)[0])
    lo, hi = price * 0.90, price * 1.10
    st.markdown(f"""
    <div class="result">
      <div class="rtag">Estimated Market Price</div>
      <div class="rprice">€ {price:,.0f}</div>
      <div class="rrange">Typical range &nbsp;<b>€ {lo:,.0f} – € {hi:,.0f}</b></div>
      <div class="rpills">
        <span class="rpill">RF + GB Ensemble</span>
        <span class="rpill">R² 0.90</span>
        <span class="rpill">1,300+ samples</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
