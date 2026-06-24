import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
import plotly.graph_objects as go

st.cache_data.clear()

# ── Cấu hình trang ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agent in CS — Analysis",
    page_icon="Logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.6,
    "axes.grid":         False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

C_PEAK   = "#00C9A7"
C_HIGH   = "#4B9CDB"
C_MID    = "#FFD93D"
C_LOW    = "#FF6B6B"
C_PURPLE = "#845EC2"
C_TEXT   = "#1a1a2e"

TIER_COLOR = {"autonomous": C_PEAK, "supervised": C_MID,
              "copilot": C_LOW,    "monitor": C_PURPLE}
TIER_BG    = {"autonomous": "#E8FBF7", "supervised": "#FFFBEA",
              "copilot": "#FFF0F0",   "monitor": "#F3EEFF"}
TIER_LABEL = {"autonomous": "Fully Autonomous", "supervised": "Supervised Agent",
              "copilot": "Co-pilot Agent",       "monitor": "Monitor Only"}
BEHAVIOR   = {
    "autonomous": "Chạy 100% tự động — chỉ alert khi lỗi",
    "supervised":  "Human checkpoint tại điểm có uncertainty cao",
    "copilot":     "AI đề xuất + giải thích — người phê duyệt cuối",
    "monitor":     "Observe & alert — tuyệt đối không tự action",
}
SHORT = {
    "Computer and Information Research Scientists": "CS Research Scientists",
    "Computer and Information Systems Managers":    "CS Info Sys. Managers",
    "Computer Network Support Specialists":         "Network Support Spec.",
    "Computer Programmers":                         "Computer Programmers",
    "Computer Systems Analysts":                    "Sys. Analysts",
    "Computer Systems Engineers/Architects":        "CS Engineers/Architects",
    "Computer User Support Specialists":            "User Support Spec.",
    "Credit Analysts":                              "Credit Analysts",
    "Credit Counselors":                            "Credit Counselors",
    "Data Entry Keyers":                            "Data Entry Keyers",
    "Database Administrators":                      "Database Admins",
    "Information Security Analysts":                "Info Security Analysts",
    "Information Technology Project Managers":      "IT Project Managers",
    "Network and Computer Systems Administrators":  "Network & Sys Admins",
    "Quality Control Systems Managers":             "QC Systems Managers",
    "Software Quality Assurance Analysts and Testers": "SQA Analysts",
    "Web Administrators":                           "Web Administrators",
    "Web Developers":                               "Web Developers",
    "Clinical Data Managers":                       "Clinical Data Managers",
}

CS_KEYWORDS = [
    "computer", "software", "data", "network", "information",
    "database", "cybersecurity", "systems", "developer", "programmer",
    "IT ", "cloud", "machine learning", "artificial intelligence",
]

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Đang đọc dữ liệu…")
def load_data():
    df_tasks   = pd.read_csv("task_statement_with_metadata.csv")
    df_workers = pd.read_csv("domain_worker_metadata.csv")
    df_desires = pd.read_csv("domain_worker_desires.csv")
    df_expert  = pd.read_csv("expert_rated_technological_capability.csv")

    def is_cs(text):
        if pd.isna(text): return False
        t = str(text).lower()
        return any(k.lower() in t for k in CS_KEYWORDS)

    cs_tasks   = df_tasks[df_tasks["Occupation (O*NET-SOC Title)"].apply(is_cs)].copy()
    cs_workers = df_workers[df_workers["Occupation (O*NET-SOC Title)"].apply(is_cs)].copy()
    cs_desires = df_desires[df_desires["Occupation (O*NET-SOC Title)"].apply(is_cs)].copy()
    cs_expert  = df_expert[df_expert["Occupation (O*NET-SOC Title)"].apply(is_cs)].copy()

    return cs_tasks, cs_workers, cs_desires, cs_expert


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
# ── Dark/Light mode toggle ───────────────────────────────────────────────────
st.markdown("## Analysis and recommendations for using AI agents "
            "in the field of Computer Science.")
st.caption("Source: O*NET CS Workers Survey")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

with st.sidebar:
    st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0e1117 !important; }
    .stApp *, [data-testid="stAppViewContainer"] * { color: #fafafa !important; }
    [data-testid="stSidebar"] { background-color: #1a1f2e !important; }
    [data-testid="stSidebar"] * { color: #fafafa !important; }
    [data-testid="stContainer"], div.stContainer { background-color: #1a1f2e !important; border-color: #444 !important; }
    .stSelectbox > div, .stSelectbox > div > div, 
    [data-baseweb="select"] > div, [data-baseweb="input"] > div { 
        background-color: #1a1f2e !important; 
        border-color: #444 !important;
    }
    [data-baseweb="select"] span, [data-baseweb="select"] div { 
        background-color: #1a1f2e !important; 
        color: #fafafa !important; 
    }
    [data-testid="stDataFrame"] { background-color: #1a1f2e !important; }
    [data-testid="stDataFrame"] * { color: #fafafa !important; }
    [data-testid="stDataFrame"] table { background-color: #1a1f2e !important; }
    [data-testid="stDataFrame"] thead tr th { 
        background-color: #2d3561 !important; 
        color: #fafafa !important;
        border-bottom: 2px solid #4a5568 !important;
    }
    [data-testid="stDataFrame"] tbody tr:nth-child(even) { background-color: #1e2535 !important; }
    [data-testid="stDataFrame"] tbody tr:nth-child(odd) { background-color: #1a1f2e !important; }
    [data-testid="stDataFrame"] tbody tr:hover { background-color: #2d3561 !important; }
    </style>
    """, unsafe_allow_html=True)

tab_tq, tab_top = st.tabs([
    "Overview",
    "Analysis",
])


# LOAD DATA
try:
    cs_tasks, cs_workers, cs_desires, cs_expert = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Không tìm thấy file CSV: {e}\n\nĐặt 4 file CSV vào cùng thư mục với app.py.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Tổng quan
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tq: 
    st.title("📊 Overview of data in the field of Computer Science")
    st.markdown("The full analysis was performed on the Computer Science subset" \
                " filtered from the original file:")

    cards = [
        ("📋", "Task metadata",   len(cs_tasks), "rows"),
        ("👷", "CS Workers",      len(cs_workers), "rows"),
        ("❤️", "Desire ratings",  len(cs_desires), "rows"),
        ("🧑‍💼", "Expert ratings", len(cs_expert), "rows"),
    ]

    col1, col2, col3, col4 = st.columns(4, gap="large")
    for col, (icon, label, value, sub) in zip([col1, col2, col3, col4], cards):
        with col:
            with st.container(border=True):
                st.markdown(f"**{icon} {label}**")
                st.markdown(f"<span style='font-size:2rem; font-weight:700'>{value:,}</span> <span style='color:gray'>{sub}</span>", unsafe_allow_html=True)

    st.divider()

    # ── Dropdown chọn bộ dữ liệu xem trước ───────────────────────────────────
    dataset_choice = st.selectbox(
        "🔍 Select the dataset to preview:",
        options=[
            "Task metadata",
            "CS Workers",
            "Desire ratings",
            "Expert ratings",
        ],
    )

    if dataset_choice == "Task metadata":
        st.markdown(f"**{len(cs_tasks):,} rows · {len(cs_tasks.columns)} columns**")
        st.dataframe(cs_tasks, use_container_width=True, height=400)

    elif dataset_choice == "CS Workers":
        st.markdown(f"**{len(cs_workers):,} rows · {len(cs_workers.columns)} columns**")
        st.dataframe(cs_workers, use_container_width=True, height=400)

    elif dataset_choice == "Desire ratings":
        st.markdown(f"**{len(cs_desires):,} rows · {len(cs_desires.columns)} columns**")
        st.dataframe(cs_desires, use_container_width=True, height=400)

    elif dataset_choice == "Expert ratings":
        st.markdown(f"**{len(cs_expert):,} rows · {len(cs_expert.columns)} columns**")
        st.dataframe(cs_expert, use_container_width=True, height=400)

    st.divider()
    n_occs = cs_workers["Occupation (O*NET-SOC Title)"].nunique()
    st.subheader(f"Occupations in the field of Computer Science:")

    occs = sorted(cs_workers["Occupation (O*NET-SOC Title)"].unique())
    occ_df = pd.DataFrame({
        "STT":        range(1, len(occs) + 1),
        "Tên nghề":   occs,
        "Số workers": [
            cs_workers[cs_workers["Occupation (O*NET-SOC Title)"] == o].shape[0]
            for o in occs
        ],
        "Số task": [
            cs_tasks[cs_tasks["Occupation (O*NET-SOC Title)"] == o].shape[0]
            if "Occupation (O*NET-SOC Title)" in cs_tasks.columns else 0
            for o in occs
        ],
    })
    st.dataframe(
        occ_df,
        use_container_width=True,
        hide_index=True,
        height=min(40 * len(occs) + 38, 600),
        column_config={
            "STT":        st.column_config.NumberColumn("STT", width=60),
            "Tên nghề":   st.column_config.TextColumn("Tên nghề", width="large"),
            "Số workers": st.column_config.NumberColumn("Số workers", width=120),
            "Số task":    st.column_config.NumberColumn("Số task", width=100),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 2.2 Top tasks
# ═══════════════════════════════════════════════════════════════════════════════
with tab_top:
    
    section_titles = {
        "Top task easy/difficult to automate": "🏆 Top Tasks: Easiest & Most Difficult to Automate",
        "Attitude & LLM Usage":               "🧠 CS Workers' Attitudes towards AI & LLM Usage",
        "Education and LLM":                  "🎓 Education affects the level of use of LLMs",
        "Expert vs Worker":                   "⚖️ Expert vs Worker: Ai đánh giá AI cao hơn?",
        "Task Categories by Industry":        "📋 Classification of AI-supported Tasks across CS Fields",
        "Experience using LLM":               "💼 Experience × LLM Usage in Computer Science",
        "Industry and LLM":                   "🔥 Heatmap: LLM Usage Frequency in Computer Science",
        "Income and LLM":                     "💰 Income affects LLM Usage in Computer Science",
        "Agent Hierarchy":                    "🗂️ Phân tầng Dynamic vs Static Agent",
    }

    section = st.selectbox(
        "📂 Chọn phần phân tích:",
        options=list(section_titles.keys()),
    )

    st.title(section_titles[section])
    st.divider()

    if section == "Top task easy/difficult to automate":
            expert_avg = (cs_expert
                .groupby(["Task ID", "Task", "Occupation (O*NET-SOC Title)"])
                .agg(
                    Score=("Automation Capacity Rating", "mean"),
                    Expert_Count=("Automation Capacity Rating", "count")
                )
                .reset_index()
                .sort_values(["Score", "Expert_Count"], ascending=[False, False]))

            top10 = expert_avg.head(10)
            bot10 = expert_avg.sort_values(["Score", "Expert_Count"], ascending=[True, False]).head(10)

            st.metric("Total number of analysis tasks", len(expert_avg))
            st.divider()

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### ✅ Top 10 easiest tasks to automate")
                st.dataframe(
                    top10[["Task", "Occupation (O*NET-SOC Title)", "Score", "Expert_Count"]]
                    .rename(columns={"Occupation (O*NET-SOC Title)": "Nghề"})
                    .reset_index(drop=True)
                    .style.format({"Score": "{:.2f}"})
                    .background_gradient(subset=["Score"], cmap="Greens"),
                    use_container_width=True, height=380,
                )
            with col_b:
                st.markdown("#### ⛔ Top 10 most difficult tasks to automate")
                st.dataframe(
                    bot10[["Task", "Occupation (O*NET-SOC Title)", "Score", "Expert_Count"]]
                    .rename(columns={"Occupation (O*NET-SOC Title)": "Nghề"})
                    .reset_index(drop=True)
                    .style.format({"Score": "{:.2f}"})
                    .background_gradient(subset=["Score"], cmap="Reds_r"),
                    use_container_width=True, height=380,
                )

            col_e, col_d = st.columns(2)
            for col, df, title, color in [
                (col_e, top10, "Top 10 easiest tasks to automate",    "#00C9A7"),
                (col_d, bot10, "Top 10 most difficult tasks to automate", "#FF6B6B"),
            ]:
                tasks_short = [t[:50] + "…" if len(t) > 50 else t for t in df["Task"]]
                fig_t = go.Figure(go.Bar(
                    x=df["Score"].values,
                    y=tasks_short,
                    orientation="h",
                    marker_color=color,
                    opacity=0.85,
                    text=[f"{v:.2f}" for v in df["Score"]],
                    textposition="outside",
                ))
                fig_t.update_layout(
                    title=title,
                    xaxis_title="Automation Capacity Rating",
                    xaxis_range=[0, 5.8],
                    height=420,
                    margin=dict(l=10, r=60, t=50, b=10),
                    showlegend=False,
                )
                fig_t.update_yaxes(autorange="reversed")
                with col:
                    st.plotly_chart(fig_t, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 2.3 Thái độ & LLM
# ═══════════════════════════════════════════════════════════════════════════════
    elif section == "Attitude & LLM Usage":

        att_cols = ["AI Tedious Work Attitude", "AI Job Importance Attitude",
                    "AI Daily Interest Attitude", "AI Suffering Attitude"]
        att_map  = {"Strongly agree": 5, "Somewhat agree": 4,
                    "Neither agree nor disagree": 3, "Somewhat disagree": 2, "Strongly disagree": 1}

        w = cs_workers.copy()
        for c in att_cols:
            w[c + "_score"] = w[c].map(att_map)

        att_df = w.groupby("Occupation (O*NET-SOC Title)")[[c + "_score" for c in att_cols]].mean()
        att_df.columns = ["Nhàm chán", "AI Quan trọng", "Quan tâm hàng ngày", "Lo ngại AI"]

        llm_cols = [c for c in w.columns if "LLM Usage by Type" in c]
        llm_map2 = {"Daily": 4, "Weekly": 3, "Monthly": 2, "Never": 1}
        for c in llm_cols:
            w[c + "_score"] = w[c].map(llm_map2)
        llm_avg = (w[[c + "_score" for c in llm_cols]].mean()
                .rename(lambda x: x.replace("_score", "").replace("LLM Usage by Type - ", "")))

        short_labels = [o.replace(" and ", " & ").replace("Specialists", "Spec.")
                        .replace("Administrators", "Admin.")[:35] for o in att_df.index]

        col_h, col_b = st.columns(2)
        with col_h:
            fig_h = px.imshow(
                att_df.values,
                x=att_df.columns.tolist(),
                y=short_labels,
                color_continuous_scale="RdYlGn",
                zmin=1, zmax=5,
                text_auto=".2f",
                title="Thái độ với AI theo Nghề nghiệp",
                aspect="auto",
                labels={"color": "Điểm TB (1–5)"},
            )
            fig_h.update_layout(height=550, margin=dict(l=10, r=10, t=50, b=10))
            fig_h.update_traces(textfont_size=10)
            st.plotly_chart(fig_h, use_container_width=True)

        with col_b:
            colors_llm = ["#4B9CDB" if v >= 3 else "#FFD93D" if v >= 2 else "#FF6B6B"
                          for v in llm_avg.values]
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=llm_avg.values, y=llm_avg.index,
                orientation="h",
                marker_color=colors_llm,
                text=[f"{v:.2f}" for v in llm_avg.values],
                textposition="outside",
            ))
            fig_b.add_vline(x=3, line_dash="dash", line_color="gray",
                            annotation_text="Weekly", annotation_position="top right")
            fig_b.add_vline(x=4, line_dash="dash", line_color="green",
                            annotation_text="Daily", annotation_position="top right")
            fig_b.update_layout(
                title="Tần suất dùng LLM theo Loại Công việc",
                xaxis_title="Tần suất TB (1=Never → 4=Daily)",
                xaxis_range=[1, 4.8],
                height=550,
                margin=dict(l=10, r=60, t=50, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_b, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 2.4 Học vấn
# ═══════════════════════════════════════════════════════════════════════════════
    elif section == "Education and LLM":

        llm_map3 = {
            "Yes, I use them every day in my work.":                    4,
            "Yes, I use them every week in my work.":                   3,
            "Yes, I have used them occasionally for specific tasks.":   2,
            "No, I have not used them for any work-related activities.": 1,
            "No, I've never heard of them.":                             1,
        }
        w = cs_workers.copy()
        w["LLM_Work_Score"] = w["LLM Use in Work"].map(llm_map3)

        edu_order = ["High School", "Some College, No Degree", "Associate Degree",
                    "Bachelor's Degree", "Master's Degree",
                    "Professional Degree (e.g., MD, JD)", "Doctorate (e.g., PhD)"]
        edu_llm = (w.groupby("Education")["LLM_Work_Score"].mean()
                .reindex([e for e in edu_order if e in w["Education"].unique()]))

        edu_colors = ["#00C9A7" if v >= 3 else "#FFD93D" if v >= 2 else "#FF6B6B"
                      for v in edu_llm.values]
        
        # Pie chart: tỷ lệ worker theo học vấn
        edu_counts = w["Education"].value_counts().reindex(
            [e for e in edu_order if e in w["Education"].unique()])
        pie_colors = ["#00C9A7", "#FFD93D", "#FF6B6B", "#74C0FC", "#B197FC", "#FFA94D", "#63E6BE"]
        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=edu_counts.index.tolist(),
            values=edu_counts.values,
            marker=dict(colors=pie_colors[:len(edu_counts)]),
            textinfo="percent",
            textfont=dict(size=11),
            hole=0.3,
        ))

        fig_pie.update_layout(
            title="Tỷ lệ Worker theo Học vấn",
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=True,
        )
        fig_edu = go.Figure()
        fig_edu.add_trace(go.Bar(
            x=edu_llm.index.tolist(),
            y=edu_llm.values,
            marker_color=edu_colors,
            text=[f"{v:.2f}" for v in edu_llm.values],
            textposition="outside",
        ))
        fig_edu.add_hline(y=3, line_dash="dash", line_color="gray",
                          annotation_text="Weekly (3)", annotation_position="right")
        fig_edu.add_hline(y=4, line_dash="dash", line_color="green",
                          annotation_text="Daily (4)", annotation_position="right")
        fig_edu.update_layout(
            title="Học vấn cao hơn → Dùng LLM nhiều hơn?",
            yaxis_title="Mức độ dùng LLM trong công việc (TB)",
            xaxis_title="",
            height=600,
            bargap=0.5,
            margin=dict(l=60, r=20, t=50, b=130),
            showlegend=False,
            plot_bgcolor="white",
            yaxis=dict(range=[0, 4.9]),
            xaxis=dict(tickangle=0, tickfont=dict(size=11)),
            annotations=[dict(
                text="1=Not used | 2=Occasionally | 3=Weekly | 4=Daily",
                xref="paper", yref="paper", x=0, y=-0.18,
                showarrow=False, font=dict(size=10, color="gray"),
            )],
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(fig_pie, width="stretch")
        with col2:
            st.plotly_chart(fig_edu, width="stretch")

        st.info("💡 **Nhận xét:** Nhóm High School và Doctorate dùng AI nhiều hơn các nhóm trung gian — "
                "có thể do High School cần AI để bù đắp thiếu hụt kỹ năng, còn Doctorate tiếp cận AI "
                "như một công cụ nghiên cứu tiên tiến.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 2.5 Expert vs Worker
# ═══════════════════════════════════════════════════════════════════════════════
    elif section == "Expert vs Worker":

        desire_avg = (cs_desires.groupby("Task ID")[
            ["Automation Desire Rating", "Human Agency Scale Rating",
            "Enjoyment Rating", "Job Security Rating"]].mean())
        expert_avg2 = (cs_expert.groupby("Task ID")[
            ["Automation Capacity Rating", "Human Agency Scale Rating"]].mean())
        compare = desire_avg.join(expert_avg2, rsuffix="_expert").dropna()
        compare["gap"] = compare["Automation Capacity Rating"] - compare["Automation Desire Rating"]

        pos = (compare["gap"] > 0).sum()
        neg = (compare["gap"] < 0).sum()

        # CSS tắt hiệu ứng rung delta
        st.markdown("""
        <style>
        [data-testid="stMetricDelta"] svg { display: none; }
        [data-testid="stMetricDelta"] { animation: none !important; }
        div[data-testid="metric-container"] > label { font-weight: 600; }
        </style>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Gap TB (Expert − Worker)", f"{compare['gap'].mean():.2f}")
        col2.metric(
            "Expert cao hơn",
            f"{pos} task",
            delta=f"{pos/len(compare):.0%}",
            delta_color="normal",
        )
        col3.metric(
            "Worker muốn cao hơn",
            f"{neg} task",
            delta=f"{neg/len(compare):.0%}",
            delta_color="normal",
        )

        st.divider()

        compare["gap_color"] = compare["gap"].apply(
            lambda g: "Expert cao hơn" if g > 0.5 else ("Worker cao hơn" if g < -0.5 else "Tương đương")
        )
        color_map_sc = {"Expert cao hơn": C_LOW, "Tương đương": C_MID, "Worker cao hơn": C_PEAK}

        col_sc, col_hi = st.columns(2)
        with col_sc:
            fig_sc = px.scatter(
                compare.reset_index(),
                x="Automation Desire Rating",
                y="Automation Capacity Rating",
                color="gap_color",
                color_discrete_map=color_map_sc,
                opacity=0.7,
                title="Scatter: Desire vs Capacity",
                labels={
                    "Automation Desire Rating": "Worker muốn tự động hóa (1–5)",
                    "Automation Capacity Rating": "Expert đánh giá AI có thể làm (1–5)",
                    "gap_color": "Phân loại",
                },
            )
            fig_sc.add_shape(type="line", x0=1, y0=1, x1=5.5, y1=5.5,
                             line=dict(dash="dash", color="gray", width=1))
            fig_sc.update_layout(
                height=480,
                margin=dict(l=20, r=20, t=60, b=20),
                plot_bgcolor="white",
                legend=dict(orientation="v", x=0.02, y=0.98, xanchor="left", yanchor="top",
                            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1),
                xaxis=dict(showgrid=True, gridcolor="lightgray", gridwidth=1),
                yaxis=dict(showgrid=True, gridcolor="lightgray", gridwidth=1),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with col_hi:
            fig_hi = px.histogram(
                compare, x="gap", nbins=20,
                title="Phân phối Khoảng chênh lệch (Expert − Worker)",
                labels={"gap": "Gap = Expert − Worker", "count": "Số nhiệm vụ"},
                color_discrete_sequence=[C_PURPLE],
                opacity=0.85,
            )
            fig_hi.update_traces(
                marker_line_color="white",
                marker_line_width=1.5,
            )
            fig_hi.add_vline(x=0, line_dash="dash", line_color="black")
            fig_hi.add_vline(x=compare["gap"].mean(), line_dash="solid",
                             line_color=C_LOW,
                             annotation_text=f"TB gap = {compare['gap'].mean():.2f}",
                             annotation_position="top right")
            fig_hi.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper",
                text=f"Expert cao hơn",
                showarrow=False, font=dict(color=C_LOW, size=11), xanchor="left")
            fig_hi.add_annotation(x=0.02, y=0.87, xref="paper", yref="paper",
                text=f"Worker muốn cao hơn",
                showarrow=False, font=dict(color=C_PEAK, size=11), xanchor="left")
            fig_hi.update_layout(
                title="Phân phối Khoảng chênh lệch",
                height=480,
                margin=dict(l=20, r=20, t=60, b=20),
                plot_bgcolor="white",
                yaxis_title="Số nhiệm vụ",
            )
            st.plotly_chart(fig_hi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 2.6 Phân loại task
# ═══════════════════════════════════════════════════════════════════════════════
    elif section == "Task Categories by Industry":

        merged26 = cs_tasks.merge(
            cs_expert[["Task ID", "Automation Capacity Rating", "Human Agency Scale Rating",
                    "Physical Action Requirement", "Involved Uncertainty",
                    "Domain Expertise Requirement", "Interpersonal Communication Requirement"]],
            on="Task ID", how="inner")

        task_avg26 = (merged26
            .groupby(["Occupation (O*NET-SOC Title)", "Task ID", "Task"])
            ["Automation Capacity Rating"].mean()
            .reset_index().round(2))

        def ai_level(score):
            if score >= 4.0: return "High"
            elif score >= 3.0: return "Medium"
            else: return "Low"

        task_avg26["AI Level"] = task_avg26["Automation Capacity Rating"].apply(ai_level)

        summary26 = (task_avg26.groupby("Occupation (O*NET-SOC Title)")
                    .agg(So_task=("Task ID", "count"),
                        AI_TB=("Automation Capacity Rating", "mean"),
                        Task_cao=("AI Level", lambda x: (x == "High").sum()),
                        Task_trung=("AI Level", lambda x: (x == "Medium").sum()),
                        Task_thap=("AI Level", lambda x: (x == "Low").sum()),)
                    .round(2).sort_values("AI_TB", ascending=False).reset_index())
        summary26.columns = ["Ngành CS", "Số Task", "AI Score TB", "Task AI cao", "Task AI TB", "Task AI thấp"]

        st.metric("Tổng số task phân tích", len(task_avg26))
        st.metric("Số ngành CS", task_avg26["Occupation (O*NET-SOC Title)"].nunique())
        st.divider()

        st.dataframe(
            summary26.style
            .background_gradient(subset=["AI Score TB"], cmap="RdYlGn", vmin=1, vmax=5)
            .format({"AI Score TB": "{:.2f}"}),
            use_container_width=True, height=500,
        )

        st.divider()
        occ_filter = st.selectbox("Xem chi tiết task của nghề:",
                                options=sorted(task_avg26["Occupation (O*NET-SOC Title)"].unique()))
        df_detail = task_avg26[task_avg26["Occupation (O*NET-SOC Title)"] == occ_filter][
            ["Task", "Automation Capacity Rating", "AI Level"]].sort_values(
            "Automation Capacity Rating", ascending=False).reset_index(drop=True)

        def color_ai_level(val):
            if "High" in str(val):   return "background-color: #00C9A7; color: white"
            elif "Medium" in str(val): return "background-color: #FFD93D; color: black"
            elif "Low" in str(val):  return "background-color: #FF6B6B; color: white"
            return ""

        st.dataframe(
            df_detail.style
            .format({"Automation Capacity Rating": "{:.2f}"})
            .background_gradient(subset=["Automation Capacity Rating"], cmap="RdYlGn", vmin=1, vmax=5)
            .map(color_ai_level, subset=["AI Level"]),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 2.7 Experience x LLM Usage
# ═══════════════════════════════════════════════════════════════════════════════
    elif section == "Experience using LLM":
        st.markdown("Phân tích mức độ sử dụng LLM theo kinh nghiệm làm việc của CS workers")

        # ── Chuẩn bị data ──────────────────────────────────────────────────────
        llm_cols27 = {
            "LLM Usage by Type - Coding":            "Coding",
            "LLM Usage by Type - System Design":     "System Design",
            "LLM Usage by Type - Data Processing":   "Data Processing",
            "LLM Usage by Type - Analysis":          "Analysis",
            "LLM Usage by Type - Information Access":"Information Access",
            "LLM Usage by Type - Idea Generation":   "Idea Generation",
            "LLM Usage by Type - Decision":          "Decision",
            "LLM Usage by Type - Communication":     "Communication",
            "LLM Usage by Type - Edit":              "Edit",
        }
        llm_freq27 = {"Daily": 4, "Weekly": 3, "Monthly": 2, "Never": 1}
        exp_order27 = ["Less than 1 year", "1-2 year", "3-5 years", "6-10 years", "More than 10 years"]

        w27 = cs_workers.copy()
        for col in llm_cols27:
            w27[col + "_score"] = w27[col].map(llm_freq27)

        score_cols27 = [c + "_score" for c in llm_cols27]
        exp_llm27 = (w27.groupby("Experience")[score_cols27].mean()
                     .reindex([e for e in exp_order27 if e in w27["Experience"].unique()]))
        exp_llm27.columns = list(llm_cols27.values())

        # ── Biểu đồ 1: Line Chart ──────────────────────────────────────────────
        st.subheader("📈 Xu hướng sử dụng LLM theo kinh nghiệm")
        fig_line27 = go.Figure()
        colors_line27 = ["#00C9A7","#4B9CDB","#FFD93D","#FF6B6B","#845EC2","#F9A825","#00B4D8","#EF476F","#06D6A0"]
        for i, col in enumerate(exp_llm27.columns):
            fig_line27.add_trace(go.Scatter(
                x=exp_llm27.index.tolist(),
                y=exp_llm27[col].values,
                mode="lines+markers",
                name=col,
                line=dict(color=colors_line27[i], width=2),
                marker=dict(size=8),
            ))
        fig_line27.add_hline(y=3, line_dash="dash", line_color="gray",
                             annotation_text="Weekly (3)", annotation_position="right")
        fig_line27.update_layout(
            xaxis_title="Kinh nghiệm làm việc",
            yaxis_title="Tần suất TB (1=Never → 4=Daily)",
            yaxis=dict(range=[1, 4.5]),
            height=480,
            margin=dict(l=60, r=20, t=40, b=60),
            plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_line27, width="stretch", key="line_exp_llm")

        st.divider()

        # ── Biểu đồ 2: Heatmap ─────────────────────────────────────────────────
        st.subheader("🔥 Heatmap: Loại LLM × Kinh nghiệm")
        fig_heat27 = px.imshow(
            exp_llm27.T.values,
            x=exp_llm27.index.tolist(),
            y=exp_llm27.columns.tolist(),
            color_continuous_scale="RdYlGn",
            zmin=1, zmax=4,
            text_auto=".2f",
            labels={"color": "Tần suất TB"},
            aspect="auto",
        )
        fig_heat27.update_layout(
            xaxis_title="Kinh nghiệm làm việc",
            yaxis_title="Loại LLM",
            height=420,
            margin=dict(l=20, r=20, t=40, b=60),
        )
        fig_heat27.update_traces(textfont_size=11)
        st.plotly_chart(fig_heat27, width="stretch", key="heat_exp_llm")

        st.info("💡 **Nhận xét:** Heatmap cho thấy loại LLM nào được dùng nhiều nhất theo từng nhóm kinh nghiệm — "
                "senior workers có xu hướng dùng LLM cho các task phức tạp hơn như System Design và Analysis.") 


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 2.8 Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
    elif section == "Industry and LLM":

        llm_types28 = {
            "LLM Usage by Type - Coding":            "Coding",
            "LLM Usage by Type - Analysis":          "Analysis",
            "LLM Usage by Type - Data Processing":   "Data Processing",
            "LLM Usage by Type - System Design":     "System Design",
            "LLM Usage by Type - Information Access":"Information",
            "LLM Usage by Type - Idea Generation":   "Idea Generation",
            "LLM Usage by Type - Decision":          "Decision",
        }
        llm_freq28 = {"Daily": 4, "Weekly": 3, "Monthly": 2, "Never": 1}
        w28 = cs_workers.copy()
        for col in llm_types28:
            w28[col + "_score"] = w28[col].map(llm_freq28)

        llm_by_occ28 = (w28.groupby("Occupation (O*NET-SOC Title)")
                        [[c + "_score" for c in llm_types28]].mean()
                        .rename(columns={c + "_score": v for c, v in llm_types28.items()}))

        merged28 = cs_tasks.merge(cs_expert[["Task ID", "Automation Capacity Rating"]], on="Task ID", how="inner")
        ai_by_occ28 = merged28.groupby("Occupation (O*NET-SOC Title)")["Automation Capacity Rating"].mean().rename("AI Score")
        combined28  = llm_by_occ28.join(ai_by_occ28).dropna().sort_values("AI Score", ascending=False)
        plot_data28 = combined28.drop(columns="AI Score")

        short_idx28 = [o.replace(" and ", " & ").replace("Specialists", "Spec.")
                       .replace("Administrators", "Admin.")[:35] for o in plot_data28.index]
        fig28 = px.imshow(
            plot_data28.values,
            x=plot_data28.columns.tolist(),
            y=short_idx28,
            color_continuous_scale="Blues",
            zmin=1, zmax=4,
            text_auto=".2f",
            title="Heatmap: Tần suất dùng LLM × Ngành CS (sắp xếp theo AI Score giảm dần)",
            aspect="auto",
            labels={"color": "Tần suất (1=Never → 4=Daily)"},
        )
        fig28.update_layout(height=580, margin=dict(l=10, r=10, t=60, b=10))
        fig28.update_traces(textfont_size=10)
        fig28.update_xaxes(tickangle=-25)
        st.plotly_chart(fig28, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 2.9 Thu nhập
# ═══════════════════════════════════════════════════════════════════════════════
    elif section == "Income and LLM":

        income_order29 = ["0-30K", "30-60K", "60-86K", "86K-165K", "165K-209K", "209K-529K", "529K+"]
        llm_work_map29 = {
            "Yes, I use them every day in my work.":                     4,
            "Yes, I use them every week in my work.":                    3,
            "Yes, I have used them occasionally for specific tasks.":    2,
            "No, I have not used them for any work-related activities.": 1,
            "No, I've never heard of them.":                             1,
        }
        familiarity_map29 = {
            "I use them regularly.":                                               3,
            "I have some experience using them.":                                  2,
            "I have heard of them but don't know much about their functionalities.": 1,
        }
        cs_w29 = cs_workers[cs_workers["Income"].isin(income_order29)].copy()
        cs_w29["LLM_Work_Score"]       = cs_w29["LLM Use in Work"].map(llm_work_map29)
        cs_w29["LLM_Familiarity_Score"] = cs_w29["LLM Familiarity"].map(familiarity_map29)

        income_stats29 = (cs_w29.groupby("Income")["LLM_Work_Score"]
                        .agg(["mean", "std", "count"]).reindex(income_order29).dropna())
        fam_stats29    = (cs_w29.groupby("Income")["LLM_Familiarity_Score"]
                        .mean().reindex(income_order29).dropna())

        col_ln, col_br = st.columns(2)
        with col_ln:
            fig29a = go.Figure()
            fig29a.add_trace(go.Scatter(
                x=income_stats29.index.tolist() + income_stats29.index.tolist()[::-1],
                y=(income_stats29["mean"] + income_stats29["std"]).tolist() +
                  (income_stats29["mean"] - income_stats29["std"]).tolist()[::-1],
                fill="toself", fillcolor="rgba(75,156,219,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="±1 Độ lệch chuẩn",
                showlegend=True,
            ))
            fig29a.add_trace(go.Scatter(
                x=income_stats29.index.tolist(),
                y=income_stats29["mean"],
                mode="lines+markers+text",
                line=dict(color=C_HIGH, width=2.5),
                marker=dict(size=10, color=C_HIGH),
                name="TB score",
                text=[f"{m:.2f}<br>(n={int(c)})" for m, c in
                      zip(income_stats29["mean"], income_stats29["count"])],
                textposition="top center",
                textfont=dict(size=10, color=C_HIGH),
                hovertemplate="%{x}: %{text}<extra></extra>",
            ))
            fig29a.add_hline(y=3, line_dash="dash", line_color="gray")
            fig29a.add_hline(y=4, line_dash="dash", line_color=C_PEAK)
            fig29a.update_layout(
                title="Mức độ dùng LLM trong công việc<br>theo Thu nhập",
                yaxis_title="LLM Work Score TB (1–4)",
                yaxis_range=[1, 4.8],
                height=480,
                margin=dict(l=60, r=20, t=80, b=80),
                xaxis_tickangle=-25,
                plot_bgcolor="white",
                legend=dict(orientation="v", x=0.02, y=0.15,
                            bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgray", borderwidth=1),
            )
            st.plotly_chart(fig29a, width="stretch", key="fig29a")

        with col_br:
            fig29b = go.Figure()
            fig29b.add_trace(go.Bar(
                x=fam_stats29.index.tolist(),
                y=fam_stats29.values,
                marker_color=C_PEAK,
                text=[f"{v:.2f}" for v in fam_stats29.values],
                textposition="outside",
                textfont=dict(size=11, color="black"),
            ))
            fig29b.add_hline(y=2, line_dash="dash", line_color="gray",
                             annotation_text="Mức 'Some experience'", annotation_position="top left",
                             annotation_font=dict(color="gray", size=10))
            fig29b.add_hline(y=3, line_dash="dash", line_color=C_PEAK,
                             annotation_text="Mức 'Use regularly'", annotation_position="top left",
                             annotation_font=dict(color=C_PEAK, size=10))
            fig29b.update_layout(
                title="Mức độ quen thuộc với LLM<br>theo Thu nhập",
                yaxis_title="LLM Familiarity Score TB (1–3)",
                yaxis_range=[0, 3.8],
                height=480,
                margin=dict(l=60, r=20, t=80, b=80),
                xaxis_tickangle=-25,
                plot_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig29b, width="stretch", key="fig29b")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Phân tầng Agent
# ═══════════════════════════════════════════════════════════════════════════════
    elif section == "Agent Hierarchy":
        st.markdown(
            "> Mức độ **Uncertainty** × **AI Capacity** xác định kiểu agent phù hợp cho từng nghề CS."
        )

        # ── Prep data ──────────────────────────────────────────────────────────────
        exp_agg_e2 = (cs_expert.groupby("Task ID")
                    .agg({"Occupation (O*NET-SOC Title)": "first", "Task": "first",
                            "Automation Capacity Rating": "mean", "Involved Uncertainty": "mean",
                            "Domain Expertise Requirement": "mean",
                            "Interpersonal Communication Requirement": "mean",
                            "Physical Action Requirement": "mean",
                            "Human Agency Scale Rating": "mean"})
                    .reset_index())
        des_agg_e2 = (cs_desires.groupby("Task ID")
                    .agg({"Automation Desire Rating": "mean", "Enjoyment Rating": "mean"})
                    .reset_index())
        merged_e2  = exp_agg_e2.merge(des_agg_e2, on="Task ID", how="inner")
        occ_e2 = (merged_e2.groupby("Occupation (O*NET-SOC Title)")
                .agg({"Automation Capacity Rating": "mean", "Involved Uncertainty": "mean",
                        "Domain Expertise Requirement": "mean", "Human Agency Scale Rating": "mean",
                        "Enjoyment Rating": "mean"})
                .round(3).reset_index())
        occ_e2.columns = ["Occupation", "cap", "unc", "dom", "agency", "enjoy"]

        UNC_CUT_E2, CAP_CUT_E2 = 2.5, 3.0

        def assign_tier_e2(row):
            if   row.cap >= 3.5 and row.unc < 2.5: return "autonomous"
            elif row.cap >= 3.0 and row.unc < 3.0: return "supervised"
            elif row.cap >= 2.5 and row.unc < 3.5: return "copilot"
            else:                                   return "monitor"

        occ_e2["tier"]  = occ_e2.apply(assign_tier_e2, axis=1)
        occ_e2["color"] = occ_e2["tier"].map(TIER_COLOR)
        occ_e2["short"] = occ_e2["Occupation"].map(lambda x: SHORT.get(x, x[:35]))

        tier_stats_e2 = (occ_e2.groupby("tier")[["cap", "unc", "dom", "agency"]]
                        .mean().round(2).reindex(["autonomous", "supervised", "copilot", "monitor"]))
        tier_stats_e2.index = [TIER_LABEL[t] for t in tier_stats_e2.index]

        # ── KPI tier count ──────────────────────────────────────────────────────────
        tier_order_display = ["autonomous", "supervised", "copilot", "monitor"]
        cols_tier = st.columns(4)
        icons = {"autonomous": "⚡", "supervised": "👁", "copilot": "🤝", "monitor": "🔔"}
        for i, t in enumerate(tier_order_display):
            n_t = (occ_e2["tier"] == t).sum()
            cols_tier[i].metric(
                label=f"{icons[t]} {TIER_LABEL[t]}",
                value=f"{n_t} nghề",
            )

        st.divider()

        # ── Filter theo tầng ───────────────────────────────────────────────────────
        tier_filter = st.multiselect(
            "Lọc theo tầng Agent:",
            options=[TIER_LABEL[t] for t in tier_order_display],
            default=[TIER_LABEL[t] for t in tier_order_display],
        )
        tier_key_filter = {TIER_LABEL[t]: t for t in tier_order_display}
        selected_tiers  = [tier_key_filter[l] for l in tier_filter]
        occ_show = occ_e2[occ_e2["tier"].isin(selected_tiers)]

        st.divider()

        # ── Detail table (filtered) ─────────────────────────────────────────────────
        st.subheader("📋 Bảng tổng hợp phân tầng Agent")
        table_e2 = (occ_show[["short", "tier", "cap", "unc", "dom", "agency"]]
                    .copy().sort_values("cap", ascending=False).reset_index(drop=True))
        table_e2.columns = ["Nghề", "Tier", "AI Capacity", "Uncertainty", "Domain Exp.", "Human Agency"]
        table_e2["Tier Label"]     = table_e2["Tier"].map(TIER_LABEL)
        table_e2["Agent Behavior"] = table_e2["Tier"].map(BEHAVIOR)

        display_e2 = table_e2[["Nghề", "Tier Label", "AI Capacity",
                                "Uncertainty", "Domain Exp.", "Human Agency", "Agent Behavior"]]

        def highlight_tier_row(row):
            t = {v: k for k, v in TIER_LABEL.items()}.get(row["Tier Label"], "monitor")
            return [f"background-color:{TIER_BG[t]}"] * len(row)

        st.dataframe(
            display_e2.style
            .apply(highlight_tier_row, axis=1)
            .format({"AI Capacity": "{:.2f}", "Uncertainty": "{:.2f}",
                    "Domain Exp.": "{:.2f}", "Human Agency": "{:.2f}"})
            .background_gradient(subset=["AI Capacity"], cmap="Greens", vmin=1, vmax=5)
            .background_gradient(subset=["Uncertainty"], cmap="Reds",   vmin=1, vmax=5),
            use_container_width=True, hide_index=True, height=500,
        )

        st.info("📌 **Kết luận:** Uncertainty < 2.5 & Capacity ≥ 3.5 → **Fully Autonomous** "
                "(Web Admin, SQA, DB Admin…). Uncertainty cao nhất → **Monitor Only** — "
                "agent chỉ observe & alert, tuyệt đối không tự thay đổi hệ thống.")