import streamlit as st
import streamlit.components.v1 as components

def show_page():

    # ── ① Hero Banner ────────────────────────────────────────────────────────
    components.html("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
      * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
      .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(255,75,75,0.3);
        border-radius: 16px;
        padding: 2.5rem 2.5rem;
        text-align: center;
      }
      .hero-tag {
        display: inline-block;
        background: rgba(255,75,75,0.15);
        border: 1px solid rgba(255,75,75,0.4);
        color: #ff4b4b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
      }
      .hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #a0c4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
      }
      .hero p { color: #8b9ab5; font-size: 0.95rem; line-height: 1.7; }
    </style>
    <div class="hero">
      <div class="hero-tag">🌡️ Extreme Heat Events in France · 2026</div>
      <h1>Conclusion &amp; Next Steps</h1>
      <p>From 15,000 deaths in 2003 to a data-driven early-warning system —<br>
      what we built, what we learned, and where we go from here.</p>
    </div>
    """, height=200)

    # ── ② Key Metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏙️ Cities Analysed",    "4",              "Paris · Lyon · Bordeaux · Marseille")
    col2.metric("📅 Data Range",         "74 yrs (35 yrs)","1950–2024 (1990–2025)")
    col3.metric("🤖 Final Model",        "XGBoost",        "Hybrid: statistical + physical")
    col4.metric("⚡ Runs on any laptop", "< 2 min",        "No GPU required")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ③ Physics of a Heatwave — 3×2 card grid ─────────────────────────────
    components.html("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
      * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
      .section-label {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; color: #ff4b4b;
        margin-bottom: 0.5rem;
      }
      .section-subtitle {
        color: #8b9ab5; font-size: 0.82rem; margin-bottom: 1.2rem; line-height: 1.6;
      }
      .section-subtitle strong { color: #e0e6f0; }
      .grid3 {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
      }
      .icard {
        background: linear-gradient(135deg, #1a1f2e, #0f3460);
        border: 1px solid rgba(160,196,255,0.15);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
      }
      .icard-icon { font-size: 1.5rem; flex-shrink: 0; margin-top: 2px; }
      .icard-title { font-size: 0.82rem; font-weight: 700; color: #a0c4ff; margin-bottom: 0.25rem; }
      .icard-text  { font-size: 0.77rem; color: #8b9ab5; line-height: 1.5; }
    </style>
    <div class="section-label">🌡️ What We Learned — The Physics of a Heatwave</div>
    <div class="section-subtitle">
      A heatwave is <strong>not just a hot day</strong> — it is a persistent atmospheric
      circulation event driven by multiple interacting mechanisms.
    </div>
    <div class="grid3">
      <div class="icard">
        <div class="icard-icon">☀️</div>
        <div>
          <div class="icard-title">Strong Solar Radiation</div>
          <div class="icard-text">Intense insolation heats the surface, which re-emits IR radiation and warms the lower atmosphere.</div>
        </div>
      </div>
      <div class="icard">
        <div class="icard-icon">🌬️</div>
        <div>
          <div class="icard-title">Very Low Wind Speed</div>
          <div class="icard-text">Stagnant air prevents heat dispersion. Wind features were the strongest predictors in both models.</div>
        </div>
      </div>
      <div class="icard">
        <div class="icard-icon">⬇️</div>
        <div>
          <div class="icard-title">Sinking Air (Anticyclone)</div>
          <div class="icard-text">Persistent high-pressure systems trap heat near the surface — the key driver of multi-day events.</div>
        </div>
      </div>
      <div class="icard">
        <div class="icard-icon">🌊</div>
        <div>
          <div class="icard-title">Gulf Stream Influence</div>
          <div class="icard-text">Weakening Atlantic circulation reduces the natural cooling buffer for Western Europe, amplifying events.</div>
        </div>
      </div>
      <div class="icard">
        <div class="icard-icon">🌡️</div>
        <div>
          <div class="icard-title">Upper-Air Heat Bubbles</div>
          <div class="icard-text">850 hPa &amp; 500 hPa anomalies build up days before surface temperatures peak — our key lag features.</div>
        </div>
      </div>
      <div class="icard">
        <div class="icard-icon">🏙️</div>
        <div>
          <div class="icard-title">Urban Amplification</div>
          <div class="icard-text">Paris shows the strongest urban heat island effect. Marseille is most exposed when the sea breeze collapses.</div>
        </div>
      </div>
    </div>
    """, height=310)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ④ Two-column: Conclusions + Next Steps ───────────────────────────────
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        components.html("""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
          * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
          .section-label {
            font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
            text-transform: uppercase; color: #00c864;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(0,200,100,0.2);
          }
          .citem {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
          }
          .citem:last-child { border-bottom: none; }
          .check {
            min-width: 20px; height: 20px;
            background: rgba(0,200,100,0.15);
            border: 1px solid rgba(0,200,100,0.4);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.6rem; color: #00c864;
            flex-shrink: 0; margin-top: 2px;
          }
          .ctitle { font-size: 0.85rem; font-weight: 700; color: #e0e6f0; margin-bottom: 0.2rem; }
          .cdesc  { font-size: 0.77rem; color: #8b9ab5; line-height: 1.5; }
        </style>
        <div class="section-label">✅ Key Conclusions</div>
        <div class="citem">
          <div class="check">✓</div>
          <div><div class="ctitle">More features = better predictions</div>
          <div class="cdesc">Expanding from wind-only (Model 1) to Copernicus reanalysis data — 850 hPa, 500 hPa, humidity, soil moisture — significantly improved forecast skill.</div></div>
        </div>
        <div class="citem">
          <div class="check">✓</div>
          <div><div class="ctitle">Statistical × Physical hybrid</div>
          <div class="cdesc">Z-score anomaly detection combined with atmospheric lag features outperforms either approach alone. Day/night Z-score separation removes seasonal noise effectively.</div></div>
        </div>
        <div class="citem">
          <div class="check">✓</div>
          <div><div class="ctitle">Regionality matters</div>
          <div class="cdesc">Same core mechanism across all four cities, but local amplifiers differ: sea breeze collapse in Marseille, urban heat island in Paris, continental blocking in Lyon.</div></div>
        </div>
        <div class="citem">
          <div class="check">✓</div>
          <div><div class="ctitle">Training cutoff prevents leakage</div>
          <div class="cdesc">Training until 2016, testing on 2016–2023 ensures a realistic evaluation. The model generalises well to unseen recent events.</div></div>
        </div>
        <div class="citem">
          <div class="check">✓</div>
          <div><div class="ctitle">Baseline clearly outperformed</div>
          <div class="cdesc">XGBoost substantially outperforms the P95 meteorological baseline in PR-AUC and F1, especially for early detection before surface temperatures peak.</div></div>
        </div>
        <div class="citem">
          <div class="check">✓</div>
          <div><div class="ctitle">Become a meteorologist</div>
          <div class="cdesc">The biggest gains came not from tuning hyperparameters, but from understanding the physical mechanisms and engineering the right features.</div></div>
        </div>
        """, height=560)

    with col_right:
        components.html("""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
          * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
          .section-label {
            font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
            text-transform: uppercase; color: #ff4b4b;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255,75,75,0.2);
          }
          .timeline { position: relative; padding-left: 1.4rem; }
          .timeline::before {
            content: '';
            position: absolute;
            left: 0.4rem; top: 0.4rem; bottom: 0;
            width: 2px;
            background: linear-gradient(to bottom, #ff4b4b 40%, rgba(255,75,75,0.1));
          }
          .titem { position: relative; margin-bottom: 1.2rem; }
          .titem:last-child { margin-bottom: 0; }
          .tdot {
            position: absolute;
            left: -1.4rem; top: 0.3rem;
            width: 13px; height: 13px;
            border-radius: 50%;
            border: 2px solid #ff4b4b;
            background: #0f1117;
          }
          .tdot.done { background: #ff4b4b; }
          .tdot.next { background: rgba(255,75,75,0.4); }
          .tstatus { font-size: 0.62rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.15rem; }
          .s-done  { color: #00c864; }
          .s-next  { color: #ff4b4b; }
          .s-later { color: #6b7a99; }
          .ttitle { font-size: 0.87rem; font-weight: 600; color: #e0e6f0; margin-bottom: 0.15rem; }
          .tdesc  { font-size: 0.76rem; color: #6b7a99; line-height: 1.5; }
        </style>
        <div class="section-label">🚀 Next Steps</div>
        <div class="timeline">
          <div class="titem">
            <div class="tdot done"></div>
            <div class="tstatus s-done">✔ Completed</div>
            <div class="ttitle">Proof of Concept — GBR Model</div>
            <div class="tdesc">Gradient Boosting on daily wind + temperature data. Validated heatwave detection for 4 French cities.</div>
          </div>
          <div class="titem">
            <div class="tdot done"></div>
            <div class="tstatus s-done">✔ Completed</div>
            <div class="ttitle">Final Model — XGBoost + Copernicus</div>
            <div class="tdesc">Hourly reanalysis data, Z-score normalisation, lag features, Heat Aridity Index. Klement event labelling.</div>
          </div>
          <div class="titem">
            <div class="tdot next"></div>
            <div class="tstatus s-next">→ Up Next</div>
            <div class="ttitle">Persistence Feature</div>
            <div class="tdesc">Add 48h rolling minimum of 850 hPa temperature to reduce false positives identified by SHAP analysis.</div>
          </div>
          <div class="titem">
            <div class="tdot next"></div>
            <div class="tstatus s-next">→ Up Next</div>
            <div class="ttitle">Expanded Spatial Coverage</div>
            <div class="tdesc">Extend to Spain, Italy, and Germany for regional-scale early warning and better model generalisation.</div>
          </div>
          <div class="titem">
            <div class="tdot"></div>
            <div class="tstatus s-later">◌ Future</div>
            <div class="ttitle">Climate-Adaptive Thresholding</div>
            <div class="tdesc">Rolling 30-year percentile window instead of fixed reference — accounts for climate non-stationarity.</div>
          </div>
          <div class="titem">
            <div class="tdot"></div>
            <div class="tstatus s-later">◌ Future</div>
            <div class="ttitle">Probabilistic Predictions</div>
            <div class="tdesc">Output calibrated probabilities instead of binary alarms, enabling risk-based decision thresholds.</div>
          </div>
          <div class="titem">
            <div class="tdot"></div>
            <div class="tstatus s-later">◌ Future</div>
            <div class="ttitle">Deep Learning Benchmark</div>
            <div class="tdesc">Evaluate LSTM / Transformer models to capture long-range temporal dependencies beyond fixed-lag features.</div>
          </div>
        </div>
        """, height=560)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ⑤ Business Value — 3×2 grid ─────────────────────────────────────────
    components.html("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
      * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
      .section-label {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; color: #00c864;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0,200,100,0.2);
      }
      .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
      .bcard {
        background: linear-gradient(135deg, #0d1f0d, #1a2e1a);
        border: 1px solid rgba(0,200,100,0.2);
        border-radius: 12px;
        padding: 1rem 1.1rem;
      }
      .bcard-icon { font-size: 1.4rem; margin-bottom: 0.5rem; }
      .bcard-title { font-size: 0.82rem; font-weight: 700; color: #00c864; margin-bottom: 0.3rem; }
      .bcard-text  { font-size: 0.77rem; color: #8b9ab5; line-height: 1.5; }
    </style>
    <div class="section-label">💼 Why This Matters — Business &amp; Societal Value</div>
    <div class="grid3">
      <div class="bcard">
        <div class="bcard-icon">💻</div>
        <div class="bcard-title">Runs on any laptop</div>
        <div class="bcard-text">Full pipeline from raw data to heatwave alarm in under 2 minutes. No GPU or cloud infrastructure required.</div>
      </div>
      <div class="bcard">
        <div class="bcard-icon">📍</div>
        <div class="bcard-title">Hyper-local precision</div>
        <div class="bcard-text">Calibrated per city — captures local amplifiers like Marseille's sea breeze collapse or Paris's urban heat island.</div>
      </div>
      <div class="bcard">
        <div class="bcard-icon">⏰</div>
        <div class="bcard-title">72h early warning</div>
        <div class="bcard-text">Upper-air lag features raise alarms before surface temperatures peak — giving authorities time to act.</div>
      </div>
      <div class="bcard">
        <div class="bcard-icon">🌍</div>
        <div class="bcard-title">Scalable across Europe</div>
        <div class="bcard-text">Modular architecture: swap in Copernicus data for any European city and retrain. Spain, Italy, Germany — same pipeline.</div>
      </div>
      <div class="bcard">
        <div class="bcard-icon">🏥</div>
        <div class="bcard-title">Direct public health impact</div>
        <div class="bcard-text">The 2003 heatwave killed 15,000 in France. A 72h alarm enables cooling centres, hospital pre-alerts, and outreach.</div>
      </div>
      <div class="bcard">
        <div class="bcard-icon">🔧</div>
        <div class="bcard-title">Regionally customisable</div>
        <div class="bcard-text">Unlike national apps with uniform thresholds, this system adapts to local climate baselines and city-specific risk profiles.</div>
      </div>
    </div>
    """, height=280)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ⑥ Quote ──────────────────────────────────────────────────────────────
    components.html("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
      * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
      .quote {
        border-left: 3px solid #ff4b4b;
        padding: 1rem 1.5rem;
        background: rgba(255,75,75,0.05);
        border-radius: 0 10px 10px 0;
        font-style: italic;
        color: #a0b0cc;
        font-size: 0.95rem;
        line-height: 1.7;
      }
      .quote span {
        display: block; margin-top: 0.5rem;
        font-style: normal; font-size: 0.78rem;
        color: #ff4b4b; font-weight: 600;
      }
    </style>
    <div class="quote">
      "To improve the model, you need to become a meteorologist. The biggest gains did not come
      from tuning hyperparameters — they came from understanding the physics and engineering the right features."
      <span>— Team Insight · Extreme Heat Project 2026</span>
    </div>
    """, height=110)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ⑦ CTA + Contact ──────────────────────────────────────────────────────
    components.html("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
      * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
      .cta {
        background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
        border: 1px solid rgba(255,75,75,0.4);
        border-radius: 14px;
        padding: 1.8rem 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
      }
      .cta h3 { font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 0.4rem; }
      .cta p  { color: #8b9ab5; font-size: 0.88rem; margin-bottom: 1rem; }
      .badges { display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; }
      .badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        color: #c8d0e0;
        font-size: 0.8rem; font-weight: 600;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        text-decoration: none;
        cursor: pointer;
      }
      .badge:hover { background: rgba(255,75,75,0.15); border-color: rgba(255,75,75,0.4); color: #ff8080; }
      .footer {
        text-align: center;
        color: #3d4a63;
        font-size: 0.72rem;
        letter-spacing: 1px;
        padding-top: 1rem;
      }
    </style>
    <div class="cta">
      <h3>🌡️ The heat is coming. The alarm is ready.</h3>
      <p>Interested in the methodology, data, or a collaboration? Feel free to reach out to the team.</p>
      <div class="badges">
        <a class="badge" href="mailto:eduard.belsch.hamburg@gmail.com">📧 Contact us</a>
        <a class="badge" href="https://github.com/VirtuallyCertain/extreme_heat" target="_blank">🔗 GitHub</a>
      </div>
    </div>
    <div class="footer">
      EXTREME HEAT ANALYSIS IN FRANCE &nbsp;·&nbsp;
      Paris · Lyon · Bordeaux · Marseille &nbsp;·&nbsp;
      Data: data.gouv.fr &amp; Copernicus &nbsp;·&nbsp; 2026
    </div>
    """, height=200)
