import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import time

# ------------------------------------------------------------------------------
# 1. ตั้งค่าหน้าเว็บ
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Japan Flight Predictor 🌸",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 2. CSS ที่แก้ไขปัญหาทั้งหมด
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mali:wght@400;500;600;700&display=swap');

    /* === พื้นหลังและฟอนต์ === */
    .stApp {
        background: linear-gradient(135deg, #FFE3EA 0%, #FFF0F5 100%) !important;
        background-attachment: fixed !important;
    }
    
    * {
        font-family: 'Mali', cursive !important;
        color: #6B4C5E !important;
    }

    /* === Header === */
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-bottom: 2px solid #FFD1DC !important;
    }
    
    header[data-testid="stHeader"] button {
        color: #6B4C5E !important;
    }
    
    footer { 
        visibility: hidden; 
    }

/* --- แก้จุดที่ 1: เปลี่ยนตัวหนังสือเป็นสัญลักษณ์ » --- */
    [data-testid="collapsedControl"] {
        font-size: 0 !important; /* ฆ่าตัวหนังสือทิ้ง */
        color: transparent !important;
        width: 40px !important;
        height: 40px !important;
        position: relative !important;
    }

    /* ใส่สัญลักษณ์ » แทนที่ตัวหนังสือ แม้ตอนชี้ก็จะเป็นสัญลักษณ์นี้ */
    [data-testid="collapsedControl"]::before {
        content: "<3"; /* นี่คือสัญลักษณ์ Symbol ที่คุณต้องการ */
        font-size: 32px !important;
        color: #6B4C5E !important; /* สีเดียวกับธีม */
        visibility: visible !important;
        display: block !important;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -55%);
    }

    /* ซ่อน SVG เดิมทิ้งไปเลยเพื่อไม่ให้ทับซ้อน */
    [data-testid="collapsedControl"] svg {
        display: none !important;
    }
    
    /* === Sidebar สวยงาม === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFD1DC 0%, #FFE3EA 100%) !important;
        border-right: 3px solid #FF9EB5 !important;
        box-shadow: 4px 0 20px rgba(255, 129, 168, 0.15) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem !important;
    }

    /* === กล่องคำถามสวยงาม === */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 2px solid #FFB3C6 !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,243,248,0.95) 100%) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 12px rgba(255, 129, 168, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 6px 20px rgba(255, 129, 168, 0.2) !important;
        transform: translateY(-2px) !important;
    }

    /* === Label สวยงาม === */
    .stMarkdown b {
        color: #D81B60 !important;
        font-size: 16px !important;
        display: block !important;
        margin-bottom: 8px !important;
    }

    /* === Select Box - เปลี่ยนจากสีดำ === */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        border: 2px solid #FFB3C6 !important;
        border-radius: 12px !important;
        color: #6B4C5E !important;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #FF9EB5 !important;
        background-color: #FFF9FA !important;
    }
    
    /* ข้อความใน select */
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #6B4C5E !important;
    }
    
    /* Dropdown menu */
    ul[role="listbox"] {
        background-color: white !important;
        border: 2px solid #FFB3C6 !important;
        border-radius: 12px !important;
    }
    
    ul[role="listbox"] li {
        color: #6B4C5E !important;
    }
    
    ul[role="listbox"] li:hover {
        background-color: #FFF0F5 !important;
        color: #D81B60 !important;
    }

    /* === Radio Buttons - ให้เห็นว่าเลือกอันไหน === */
    div[role="radiogroup"] {
        gap: 8px !important;
    }
    
    div[role="radiogroup"] label {
        background-color: white !important;
        border: 2px solid #FFB3C6 !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        margin: 4px 0 !important;
        transition: all 0.3s ease !important;
        color: #6B4C5E !important;
        cursor: pointer !important;
    }
    
    div[role="radiogroup"] label:hover {
        border-color: #FF9EB5 !important;
        background-color: #FFF9FA !important;
    }
    
    /* เมื่อเลือกแล้ว */
    div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #FFE3EA 0%, #FFD1DC 100%) !important;
        border-color: #FF6B9D !important;
        border-width: 3px !important;
        font-weight: 600 !important;
    }
    
    /* จุดกลม radio */
    div[role="radiogroup"] label > div:first-child {
        border: 2px solid #FFB3C6 !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    div[role="radiogroup"] label[data-checked="true"] > div:first-child {
        border-color: #FF6B9D !important;
        background-color: #FF6B9D !important;
    }
    
    div[role="radiogroup"] label[data-checked="true"] > div:first-child > div {
        background-color: white !important;
    }
    
    /* ข้อความใน radio */
    div[role="radiogroup"] label span,
    div[role="radiogroup"] label p {
        color: #6B4C5E !important;
    }
    
    div[role="radiogroup"] label[data-checked="true"] span,
    div[role="radiogroup"] label[data-checked="true"] p {
        color: #D81B60 !important;
        font-weight: 600 !important;
    }

    /* === Slider === */
    div[data-testid="stSlider"] > div > div > div {
        background-color: #FFD1DC !important;
    }
    
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #FF6B9D !important;
        width: 24px !important;
        height: 24px !important;
        border: 3px solid white !important;
        box-shadow: 0 2px 8px rgba(255, 107, 157, 0.3) !important;
    }

/* === เปลี่ยนปุ่มเป็นสีชมพูเข้ม (Dark Pink) === */
    div.stButton > button {
        /* เปลี่ยนจาก Gradient อ่อน เป็นสีชมพูเข้ม #D81B60 */
        background: linear-gradient(135deg, #D81B60 0%, #AD1457 100%) !important;
        color: #FFFFFF !important;
        border-radius: 50px !important;
        height: 56px !important;
        width: 100% !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        /* ปรับเงาให้เข้ากับสีเข้ม */
        box-shadow: 0 6px 20px rgba(216, 27, 96, 0.4) !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    /* เอฟเฟกต์ตอนเอาเมาส์ไปชี้ ให้เข้มขึ้นอีกนิด */
    div.stButton > button:hover {
        background: linear-gradient(135deg, #AD1457 0%, #880E4F 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(216, 27, 96, 0.5) !important;
    }

    /* === Hero Section === */
    .hero-container {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,243,248,0.9) 100%);
        border-radius: 30px;
        margin: 20px auto 40px;
        max-width: 900px;
        box-shadow: 0 10px 40px rgba(255, 129, 168, 0.15);
        border: 3px solid #FFD1DC;
    }
    
    .hero-title {
        font-size: 52px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #D81B60 0%, #FF6B9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 16px !important;
    }
    
    .hero-subtitle {
        font-size: 18px !important;
        color: #8B6B7A !important;
        margin-bottom: 20px !important;
    }
    
    /* === GIF Container === */
    .gif-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px auto;
    }
    
    .gif-container img {
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(107, 76, 94, 0.15);
    }

    /* === Boarding Pass Card === */
    .ticket-card {
        background: white;
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(107, 76, 94, 0.15);
        display: flex;
        overflow: hidden;
        margin: 40px auto;
        max-width: 850px;
        border: 3px solid #FFD1DC;
        position: relative;
    }
    
    .ticket-left {
        padding: 48px;
        flex: 1.8;
        border-right: 3px dashed #FFD1DC;
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF9FA 100%);
    }
    
    .ticket-right {
        padding: 48px;
        flex: 1;
        background: linear-gradient(135deg, #FFE3EA 0%, #FFD1DC 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
    }
    
    .ticket-header {
        display: flex;
        align-items: center;
        margin-bottom: 30px;
    }
    
    .ticket-logo {
        font-size: 36px;
        margin-right: 12px;
    }
    
    .ticket-title {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #D81B60 !important;
        margin: 0 !important;
    }
    
    .ticket-info-row {
        display: flex;
        justify-content: space-between;
        margin: 24px 0;
        gap: 20px;
    }
    
    .ticket-info-item {
        flex: 1;
    }
    
    .ticket-label {
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8B6B7A !important;
        margin-bottom: 6px !important;
        font-weight: 600 !important;
    }
    
    .ticket-value {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #2D1F29 !important;
    }
    
    .price-container {
        text-align: center;
    }
    
    .price-label {
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6B4C5E !important;
        margin-bottom: 12px !important;
        font-weight: 600 !important;
    }
    
    .price-value {
        font-size: 56px !important;
        font-weight: 900 !important;
        color: #D81B60 !important;
        line-height: 1 !important;
        margin: 16px 0 !important;
    }
    
    .price-subtitle {
        font-size: 13px !important;
        color: #8B6B7A !important;
        font-weight: 500 !important;
    }

    /* === Route Display === */
    .route-display {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 30px 0;
        padding: 24px;
        background: linear-gradient(90deg, rgba(255,209,220,0.2) 0%, rgba(255,209,220,0.4) 50%, rgba(255,209,220,0.2) 100%);
        border-radius: 16px;
    }
    
    .airport-code {
        font-size: 36px !important;
        font-weight: 900 !important;
        color: #D81B60 !important;
    }
    
    .route-arrow {
        font-size: 28px !important;
        margin: 0 20px;
        color: #FF9EB5 !important;
    }

    /* === Animations === */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .ticket-card {
        animation: fadeInUp 0.6s ease-out;
    }

    /* === Info Alert === */
    div[data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(230,240,255,0.95) 100%) !important;
        border-left: 4px solid #4FC3F7 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    
    div[data-testid="stAlert"] p {
        color: #1976D2 !important;
        font-size: 16px !important;
    }

    /* === Spinner === */
    div[data-testid="stSpinner"] > div {
        border-top-color: #FF6B9D !important;
    }

    /* === ซ่อนส่วนที่ไม่จำเป็น === */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* === ป้องกันโค้ดแสดงใน markdown === */
    div[data-testid="stMarkdownContainer"] code {
        display: none !important;
    }
    
    div[data-testid="stMarkdownContainer"] pre {
        display: none !important;
    }

    
    
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. โหลดโมเดล
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, '..', 'models', 'flight_price_model.pkl')
    col_path = os.path.join(curr_dir, '..', 'models', 'model_columns.pkl')
    if not os.path.exists(model_path): 
        return None, None
    return joblib.load(model_path), joblib.load(col_path)

model, model_columns = load_model()

# ------------------------------------------------------------------------------
# 4. Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    
    # Totoro GIF
    st.markdown("""
    <div style='text-align: center; margin: 0 auto 24px; padding: 16px;'>
        <img src='https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExbHlvbmV1MndmNXQ4dWk3bGpwNG1kM2EyeGw5d2hsY3l1aDVpMTh2NSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/1msHrXC3xHduwy89Ze/giphy.gif' 
             style='width: 200px; border-radius: 16px;'>
    </div>
    """, unsafe_allow_html=True)
    
    

    with st.form("prediction_form"):
        # ✈️ เลือกสายการบิน
        with st.container(border=True):
            st.markdown("<b>✈️ สายการบิน</b>", unsafe_allow_html=True)
            airline = st.selectbox(
                "Airline",
                ['Thai AirAsia X', 'Zipair', 'Thai Vietjet', 'Thai Airways', 'ANA', 'JAL'],
                label_visibility="collapsed"
            )

        # 🎫 เลือกรูปแบบการบิน
        with st.container(border=True):
            st.markdown("<b>🎫 รูปแบบการบิน</b>", unsafe_allow_html=True)
            flight_type = st.radio(
                "Flight Type",
                ['Direct (บินตรง)', 'Transit (ต่อเครื่อง)'],
                label_visibility="collapsed"
            )

        # 🗓️ เลือกวันเดินทาง
        with st.container(border=True):
            st.markdown("<b>🗓️ วันเดินทาง</b>", unsafe_allow_html=True)
            day_name = st.selectbox(
                "Day",
                ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                label_visibility="collapsed"
            )

        # ⏳ จองล่วงหน้า
        with st.container(border=True):
            st.markdown("<b>⏳ จองล่วงหน้า</b>", unsafe_allow_html=True)
            days_to_flight = st.slider(
                "Days", 
                1, 180, 30,
                label_visibility="collapsed"
            )
            st.caption(f"📅 อีก {days_to_flight} วันจะเดินทางแล้ว")
        
        submit_button = st.form_submit_button("💖 คำนวณราคาเลย!")

# ------------------------------------------------------------------------------
# 5. Main Content
# ------------------------------------------------------------------------------

# Hero Section

st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Japan Flight Predictor 🌸</h1>
    <p class="hero-subtitle">ทำนายราคาตั๋วเครื่องบินไปญี่ปุ่นได้แม่นยำ พร้อมวางแผนการเดินทางของคุณ</p>
</div>
""", unsafe_allow_html=True)

# ส่วนรูป Shin-chan
st.markdown("""
<div class='gif-container'>
    <img src='https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXozOWN5NHpxdXR0MWV3NGZjYnpnaTF3bDhpaDc2cGk1dzQyanJuNiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/8Bl2ai65p7taJtpznv/giphy.gif' 
         style='width: 320px; border-radius: 16px; box-shadow: 0 8px 24px rgba(107, 76, 94, 0.15);'>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. แสดงผลลัพธ์
# ------------------------------------------------------------------------------
if submit_button:
    if model is None:
        st.error("❌ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่าไฟล์ model อยู่ในตำแหน่งที่ถูกต้อง")
    else:
        with st.spinner('🎯 กำลังคำนวณราคาตั๋วที่เหมาะสมที่สุดสำหรับคุณ...'):
            time.sleep(1.2)
            
            # เตรียมข้อมูล
            flight_type_eng = 'Direct' if 'Direct' in flight_type else 'Transit'
            input_data = pd.DataFrame({
                'Airline': [airline],
                'Flight_Type': [flight_type_eng],
                'Days_to_Flight': [days_to_flight],
                'Day_Name': [day_name]
            })
            input_prepared = pd.get_dummies(input_data).reindex(columns=model_columns, fill_value=0)
            price = model.predict(input_prepared)[0]

# --- ส่วนของ Boarding Pass (ปรับแก้ให้เหลืออันเดียวและกึ่งกลาง) ---
            import streamlit.components.v1 as components
            
            boarding_pass_html = f"""
            <link href="https://fonts.googleapis.com/css2?family=Mali:wght@400;700&display=swap" rel="stylesheet">
            <style>
                .ticket-body {{
                    font-family: 'Mali', cursive;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 10px;
                }}
                .ticket {{
                    background: white;
                    width: 650px;
                    height: 260px;
                    border-radius: 25px;
                    border: 3px solid #FFD1DC;
                    display: flex;
                    overflow: hidden;
                    box-shadow: 0 10px 30px rgba(107, 76, 94, 0.1);
                    position: relative;
                }}
                .left {{
                    flex: 1.8;
                    padding: 25px;
                    border-right: 3px dashed #FFD1DC;
                    position: relative;
                }}
                .right {{
                    flex: 1;
                    padding: 25px;
                    background: #FFF9FA;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    text-align: center;
                }}
                .notch {{
                    position: absolute;
                    width: 26px;
                    height: 26px;
                    background: #FFE3EA;
                    border-radius: 50%;
                    right: -15px;
                    z-index: 10;
                }}
                .notch-top {{ top: -13px; border-bottom: 3px solid #FFD1DC; }}
                .notch-bottom {{ bottom: -13px; border-top: 3px solid #FFD1DC; }}
                
                .ticket-title {{ color: #D81B60; font-size: 20px; font-weight: 700; margin-bottom: 15px; }}
                .route {{ font-size: 28px; font-weight: 700; color: #6B4C5E; margin-bottom: 15px; }}
                .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
                .label {{ color: #8B6B7A; font-size: 10px; font-weight: 600; }}
                .val {{ color: #6B4C5E; font-size: 15px; font-weight: 700; }}
                .price-val {{ color: #D81B60; font-size: 42px; font-weight: 800; margin: 5px 0; }}
            </style>
            <div class="ticket-body">
                <div class="ticket">
                    <div class="left">
                        <div class="notch notch-top"></div>
                        <div class="notch notch-bottom"></div>
                        <div class="ticket-title">✈️ BOARDING PASS</div>
                        <div class="route">BKK <span style="color:#FF9EB5">✈</span> TOKYO</div>
                        <div class="info-grid">
                            <div><div class="label">AIRLINE</div><div class="val">{airline}</div></div>
                            <div><div class="label">FLIGHT TYPE</div><div class="val">{flight_type_eng}</div></div>
                            <div><div class="label">DEPARTURE DAY</div><div class="val">{day_name}</div></div>
                            <div><div class="label">BOOK AHEAD</div><div class="val">{days_to_flight} Days</div></div>
                        </div>
                    </div>
                    <div class="right">
                        <div style="color: #6B4C5E; font-size: 13px; font-weight: 600;">ESTIMATED PRICE</div>
                        <div class="price-val">฿{price:,.0f}</div>
                        <div style="font-size: 11px; color: #8B6B7A;">One-way Ticket</div>
                    </div>
                </div>
            </div>
            """
            
            # แสดงผลแค่ที่เดียว
            components.html(boarding_pass_html, height=300)
            
            st.balloons()
            
            # คำแนะนำแบบกระชับ
            st.markdown(f"""
            <div style='text-align: center; margin-top: 10px; padding: 15px; background: rgba(255,255,255,0.6); border-radius: 12px; border: 2px solid #FFD1DC;'>
                <p style='color: #6B4C5E !important; font-size: 14px; margin: 0;'>
                     <b>💡Tip:</b> ราคานี้ทำนายโดย AI ลองเปลี่ยนวันเดินทางเพื่อเช็กราคาที่ถูกกว่าได้นะคะ!
                </p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("👈 เลือกข้อมูลการเดินทางของคุณจากแถบด้านซ้าย แล้วกดปุ่มหัวใจชมพูเพื่อทำนายราคา")