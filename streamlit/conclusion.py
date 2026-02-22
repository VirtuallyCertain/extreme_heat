import streamlit as st

# ── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(255,75,75,0.3);
    border-radius: 16px;
    padding: 3rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
}
.hero-tag {
    display: inline-block;
    background: rgba(255,75,75,0.15);
    border: 1px solid rgba(255,75,75,0.4);
    color: #ff4b4b;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff, #a0c4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
}
.hero p { color: #8b9ab5; font-size: 1rem; line-height: 1.7; }

.card {
    background: #1e2130;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #ff4b4b;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255,75,75,0.2);
}

.conclusion-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #c8d0e0;
    font-size: 0.92rem;
    line-height: 1.5;
}
.conclusion-item:last-child { border-bottom: none; }
.check {
    width: 20px; height: 20px;
    background: rgba(0,200,100,0.15);
    border: 1px solid rgba(0,200,100,0.4);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 0.65rem;
    color: #00c864;
    margin-top: 2px;
    padding: 2px;
}

.timeline { position: relative; padding-left: 1.5rem; }
.timeline::before {
    content: '';
    position: absolute;
    left: 0.45rem; top: 0; bottom: 0;
    width: 2px;
    background: linear-gradient(to bottom, #ff4b4b, rgba(255,75,75,0.1));
}
.timeline-item { position: relative; margin-bottom: 1.3rem; }
.timeline-item:last-child { margin-bottom: 0; }
.timeline-dot {
    position: absolute;
    left: -1.5rem; top: 0.25rem;
    width: 14px; height: 14px;
    border-radius: 50%;
    border: 2px solid #ff4b4b;
    background: #0f1117;
}
.timeline-dot.done { background: #ff4b4b; }
.timeline-status {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.status-done  { color: #00c864; }
.status-wip   { color: #f0a500; }
.status-next  { color: #ff4b4b; }
.status-later { color: #6b7a99; }
.timeline-title { font-size: 0.9rem; font-weight: 600; color: #e0e6f0; margin-bottom: 0.2rem; }
.timeline-desc  { font-size: 0.78rem; color: #6b7a99; line-height: 1.5; }

.quote-block {
    border-left: 3px solid #ff4b4b;
    padding: 1rem 1.5rem;
    background: rgba(255,75,75,0.05);
    border-radius: 0 10px 10px 0;
    font-style: italic;
    color: #a0b0cc;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-bottom: 1.5rem;
}
.quote-block span {
    display: block; margin-top: 0.5rem;
    font-style: normal; font-size: 0.78rem;
    color: #ff4b4b; font-weight: 600;
}

.cta {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 1px solid rgba(255,75,75,0.4);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.cta h3 { font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 0.4rem; }
.cta p  { color: #8b9ab5; font-size: 0.88rem; }

.footer-note {
    text-align: center;
    color: #3d4a63;
    font-size: 0.75rem;
    letter-spacing: 1px;
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


def show_page():
# ── ① Hero Banner ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">📋 Section 5 of 5</div>
        <h1>Conclusion & Next Steps</h1>
        <p>A summary of what we built, what we learned, and where we go from here —<br>turning insights into action.</p>
    </div>
    """, unsafe_allow_html=True)


# ── ② Key Metrics ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Model Accuracy",     "94%",   "+12% vs. baseline")
    col2.metric("⚡ Processing Speed",   "3×",    "faster than before")
    col3.metric("💶 Annual Savings",     "€120k", "estimated ROI")


    st.markdown("<br>", unsafe_allow_html=True)


# ── ③ Two-column: Conclusions + Next Steps ───────────────────────────────────
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        conclusions = [
            "The proposed solution significantly outperforms the existing baseline across all KPIs.",
            "Data quality was the single biggest lever — cleaning pipelines alone improved accuracy by 12%.",
            "Stakeholder feedback confirmed strong alignment with business requirements.",
            "The architecture is modular and ready for production-grade scaling.",
            "Proof of concept validated — ROI positive within the first quarter of deployment.",
        ]
        items_html = "".join(
            f'<div class="conclusion-item"><span class="check">✓</span><span>{c}</span></div>'
            for c in conclusions
        )
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Key Conclusions</div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        steps = [
            ("done",  "Proof of Concept",   "Validated core hypothesis with internal dataset and stakeholder sign-off."),
            ("wip",   "Pilot Deployment",   "Rolling out to 2 business units; collecting real-world feedback."),
            ("next",  "Full Integration",   "Connect to production systems, automate retraining pipeline."),
            ("later", "Scale & Optimize",   "Expand to all regions, monitor KPIs, iterate on model performance."),
        ]
        status_labels = {
            "done":  ("✔ Completed",   "status-done"),
            "wip":   ("⟳ In Progress", "status-wip"),
            "next":  ("→ Up Next",     "status-next"),
            "later": ("◌ Planned",     "status-later"),
        }
        timeline_html = ""
        for status, title, desc in steps:
            label, css = status_labels[status]
            dot_class = "done" if status == "done" else ""
            timeline_html += f"""
            <div class="timeline-item">
                <div class="timeline-dot {dot_class}"></div>
                <div class="timeline-status {css}">{label}</div>
                <div class="timeline-title">{title}</div>
                <div class="timeline-desc">{desc}</div>
            </div>"""

        st.markdown(f"""
        <div class="card">
            <div class="card-title">🚀 Next Steps</div>
            <div class="timeline">{timeline_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ── ④ Impact Visual (Platzhalter) ────────────────────────────────────────────
    with st.expander("📊 Impact Overview", expanded=True):
        st.info("👉 Hier kommt dein Chart rein — z. B. `st.plotly_chart(fig)` oder `st.image('diagram.png')`")
        img_col1, img_col2 = st.columns(2)
        img_col1.image("https://placehold.co/600x300/1a1f2e/ff4b4b?text=Before+%2F+After", width='content')
        img_col2.image("https://placehold.co/600x300/1a1f2e/a0c4ff?text=Architecture+Diagram", width='content')


    st.markdown("<br>", unsafe_allow_html=True)


# ── ⑤ Quote ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="quote-block">
        "The goal is not to predict the future — it's to build the capability to shape it."
        <span>— Project Team, 2025</span>
    </div>
    """, unsafe_allow_html=True)


# ── ⑥ CTA ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cta">
        <h3>Ready to move forward?</h3>
        <p>Let's align on priorities, assign owners, and set the first milestone together.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📅 Schedule Follow-up", width='content', type="primary"):
        st.success("✅ Follow-up request sent!")


# ── Footer ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="footer-note">PROJECT NAME · CONFIDENTIAL · 2025</div>', unsafe_allow_html=True)
