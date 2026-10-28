import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import requests

# -------------------------------------------------
# PASTE YOUR OPENROUTER API KEY HERE
# -------------------------------------------------
OPENROUTER_API_KEY = "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"

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Medical Device Safety System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sample Data Generator
# -------------------------------------------------
def generate_sample_data(n=40):
    devices = ["Infusion Pump", "Ventilator", "Patient Monitor", "Defibrillator", 
               "Syringe Pump", "ECG Machine", "Anesthesia Machine", "Dialysis Machine"]
    failure_types = ["Hardware Failure", "Software Error", "User Error", 
                     "Design Issue", "Maintenance Related", "Power/Battery Issue"]
    severities = ["Low", "Medium", "High", "Critical"]
    locations = ["ICU", "OT", "Emergency", "Ward", "NICU", "Dialysis Unit"]

    data = []
    for i in range(n):
        device = random.choice(devices)
        ftype = random.choice(failure_types)
        severity = random.choice(severities)
        location = random.choice(locations)

        descriptions = {
            "Hardware Failure": f"{device} stopped working during use. Possible component failure.",
            "Software Error": f"{device} displayed error code and froze. Software glitch suspected.",
            "User Error": f"Incorrect settings entered on {device} by staff leading to wrong delivery.",
            "Design Issue": f"Confusing interface on {device} caused wrong parameter selection.",
            "Maintenance Related": f"{device} failed due to overdue preventive maintenance.",
            "Power/Battery Issue": f"{device} battery drained unexpectedly during critical use."
        }

        data.append({
            "Report_ID": f"MDR-{1000+i}",
            "Date": (datetime.now() - pd.Timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d"),
            "Device": device,
            "Location": location,
            "Description": descriptions[ftype],
            "Failure_Type": ftype,
            "Severity": severity,
            "Reported_By": random.choice(["Nurse", "Doctor", "Biomedical Engineer", "Technician"]),
            "Hospital": random.choice(["City General Hospital", "Metro Care Hospital"])
        })
    return pd.DataFrame(data)

# -------------------------------------------------
# AI Classifier
# -------------------------------------------------
@st.cache_resource
def train_classifier():
    train_df = generate_sample_data(120)
    
    failure_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    failure_pipe.fit(train_df["Description"], train_df["Failure_Type"])

    severity_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=300, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    severity_pipe.fit(train_df["Description"], train_df["Severity"])

    return failure_pipe, severity_pipe

failure_model, severity_model = train_classifier()

def classify_report(description: str):
    failure_type = failure_model.predict([description])[0]
    severity = severity_model.predict([description])[0]

    device_keywords = {
        "Infusion Pump": ["infusion", "pump", "iv", "drip"],
        "Ventilator": ["ventilator", "vent", "respiration", "breathing"],
        "Patient Monitor": ["monitor", "vital", "spo2", "heart rate"],
        "Defibrillator": ["defibrillator", "defib", "shock", "aed"],
        "Syringe Pump": ["syringe", "syringe pump"],
        "ECG Machine": ["ecg", "ekg", "electrocardiogram"],
        "Anesthesia Machine": ["anesthesia", "anaesthesia"],
        "Dialysis Machine": ["dialysis", "hemodialysis"]
    }

    detected_device = "Unknown Device"
    desc_lower = description.lower()
    for device, keywords in device_keywords.items():
        if any(kw in desc_lower for kw in keywords):
            detected_device = device
            break

    return detected_device, failure_type, severity

# -------------------------------------------------
# Session State
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "hospital_name" not in st.session_state:
    st.session_state.hospital_name = ""
if "reports" not in st.session_state:
    st.session_state.reports = generate_sample_data(30)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am AshwinAI, your support assistant for the Medical Device Safety System. How can I help you today?"}
    ]

# -------------------------------------------------
# AshwinAI using OpenRouter
# -------------------------------------------------
def get_ashwin_response(user_message: str):
    system_prompt = """
You are AshwinAI, the official support assistant of the Medical Device Safety Reporting and Analysis System.

About the system:
- This system helps hospitals report and analyze safety incidents related to medical devices.
- Main features: Report Incident, Dashboard with analytics, All Reports (view and delete records).
- The system uses AI to automatically classify Failure Type and Severity.
- Two demo hospitals are available: City General Hospital and Metro Care Hospital.

Your personality:
- Friendly but professional
- Speak in simple and clear English
- You can use basic biomedical terms when needed
- Give short, practical and helpful answers
"""

    try:
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Medical Device Safety System"
        }

        payload = {
            "model": "openrouter/free",   # Free model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.5,
            "max_tokens": 500
        }

        print("\n---------- OpenRouter Request ----------")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:400]}")
        print("----------------------------------------\n")

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"Error from OpenRouter ({response.status_code}). Check Terminal."

    except Exception as e:
        print("\n---------- EXCEPTION ----------")
        print(str(e))
        print("-------------------------------\n")
        return "Connection error. Please check the Terminal."

# -------------------------------------------------
# Login Page
# -------------------------------------------------
def login_page():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Medical Device Safety Reporting System</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>Hospital Login Portal</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1.4, 1])

    with col2:
        st.markdown("##### Login")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("Login", use_container_width=True, type="primary"):
            if username == "cityhospital" and password == "city123":
                st.session_state.logged_in = True
                st.session_state.hospital_name = "City General Hospital"
                st.rerun()
            elif username == "metrocare" and password == "metro123":
                st.session_state.logged_in = True
                st.session_state.hospital_name = "Metro Care Hospital"
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown("---")
        st.caption("**Demo Accounts**")
        st.caption("City General Hospital → cityhospital / city123")
        st.caption("Metro Care Hospital → metrocare / metro123")

# -------------------------------------------------
# Main Application
# -------------------------------------------------
def main_app():
    with st.sidebar:
        st.markdown(f"### {st.session_state.hospital_name}")
        st.caption("Medical Device Safety System")
        st.markdown("---")

        menu = st.radio(
            "Navigation",
            ["Report Incident", "Dashboard", "All Reports"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.hospital_name = ""
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Hello! I am AshwinAI, your support assistant for the Medical Device Safety System. How can I help you today?"}
            ]
            st.rerun()

        st.markdown("---")
        st.markdown("### AshwinAI Support")

        chat_container = st.container(height=320)
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**AshwinAI:** {msg['content']}")

        user_input = st.chat_input("Ask AshwinAI...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("AshwinAI is typing..."):
                response = get_ashwin_response(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    # Report Incident
    if menu == "Report Incident":
        st.markdown("## Report Medical Device Incident")
        st.caption(f"Logged in as: {st.session_state.hospital_name}")
        st.markdown("---")

        with st.form("report_form", clear_on_submit=True):
            col1, col2 = st.columns(2)

            with col1:
                device = st.selectbox("Device Involved", 
                    ["Infusion Pump", "Ventilator", "Patient Monitor", "Defibrillator",
                     "Syringe Pump", "ECG Machine", "Anesthesia Machine", "Dialysis Machine", "Other"])
                location = st.selectbox("Location", 
                    ["ICU", "OT", "Emergency", "Ward", "NICU", "Dialysis Unit", "Other"])

            with col2:
                reported_by = st.selectbox("Reported By", 
                    ["Nurse", "Doctor", "Biomedical Engineer", "Technician", "Other"])
                date = st.date_input("Date of Incident", datetime.now())

            description = st.text_area("Incident Description", height=130,
                placeholder="Describe what happened with the medical device...")

            submitted = st.form_submit_button("Submit Report", use_container_width=True)

            if submitted:
                if description.strip() == "":
                    st.error("Please enter the incident description.")
                else:
                    ai_device, ai_failure, ai_severity = classify_report(description)

                    new_report = {
                        "Report_ID": f"MDR-{random.randint(2000,9999)}",
                        "Date": date.strftime("%Y-%m-%d"),
                        "Device": device if device != "Other" else ai_device,
                        "Location": location,
                        "Description": description,
                        "Failure_Type": ai_failure,
                        "Severity": ai_severity,
                        "Reported_By": reported_by,
                        "Hospital": st.session_state.hospital_name
                    }

                    st.session_state.reports = pd.concat(
                        [st.session_state.reports, pd.DataFrame([new_report])],
                        ignore_index=True
                    )

                    st.success("Report submitted successfully.")
                    st.info(f"AI Classification → Device: {new_report['Device']} | Failure Type: {ai_failure} | Severity: {ai_severity}")

    # Dashboard
    elif menu == "Dashboard":
        st.markdown("## Safety Analytics Dashboard")
        st.caption(f"Hospital: {st.session_state.hospital_name}")
        st.markdown("---")

        df = st.session_state.reports
        df = df[df["Hospital"] == st.session_state.hospital_name]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Reports", len(df))
        m2.metric("Critical Incidents", len(df[df["Severity"] == "Critical"]) if len(df) > 0 else 0)
        m3.metric("High Severity", len(df[df["Severity"] == "High"]) if len(df) > 0 else 0)
        m4.metric("Devices Involved", df["Device"].nunique() if len(df) > 0 else 0)

        st.markdown("---")

        if len(df) > 0:
            c1, c2 = st.columns(2)

            with c1:
                fig1 = px.pie(df, names="Failure_Type", title="Failure Type Distribution",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
                fig1.update_layout(margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig1, use_container_width=True)

            with c2:
                severity_count = df["Severity"].value_counts().reset_index()
                severity_count.columns = ["Severity", "Count"]
                fig2 = px.bar(severity_count, x="Severity", y="Count", title="Severity Distribution",
                              color="Severity", color_discrete_sequence=px.colors.sequential.Reds)
                fig2.update_layout(margin=dict(t=40, b=20, l=20, r=20), showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            c3, c4 = st.columns(2)

            with c3:
                device_counts = df["Device"].value_counts().reset_index()
                device_counts.columns = ["Device", "Count"]
                fig3 = px.bar(device_counts, x="Device", y="Count", title="Incidents by Device",
                              color="Count", color_continuous_scale="Blues")
                fig3.update_layout(margin=dict(t=40, b=20, l=20, r=20), xaxis_tickangle=-25)
                st.plotly_chart(fig3, use_container_width=True)

            with c4:
                loc_counts = df["Location"].value_counts().reset_index()
                loc_counts.columns = ["Location", "Count"]
                fig4 = px.pie(loc_counts, names="Location", values="Count", title="Incidents by Location",
                              color_discrete_sequence=px.colors.qualitative.Set2)
                fig4.update_layout(margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown("### Recent High & Critical Incidents")
            critical_df = df[df["Severity"].isin(["Critical", "High"])].sort_values("Date", ascending=False).head(8)
            st.dataframe(
                critical_df[["Report_ID", "Date", "Device", "Failure_Type", "Severity", "Location"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No reports available for this hospital yet.")

    # All Reports
    elif menu == "All Reports":
        st.markdown("## All Submitted Reports")
        st.caption(f"Hospital: {st.session_state.hospital_name}")
        st.markdown("---")

        df = st.session_state.reports
        df = df[df["Hospital"] == st.session_state.hospital_name].reset_index(drop=True)
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)

        f1, f2, f3 = st.columns(3)
        with f1:
            filter_severity = st.multiselect("Severity", sorted(df["Severity"].unique()) if len(df) > 0 else [])
        with f2:
            filter_device = st.multiselect("Device", sorted(df["Device"].unique()) if len(df) > 0 else [])
        with f3:
            filter_failure = st.multiselect("Failure Type", sorted(df["Failure_Type"].unique()) if len(df) > 0 else [])

        filtered = df.copy()
        if filter_severity:
            filtered = filtered[filtered["Severity"].isin(filter_severity)]
        if filter_device:
            filtered = filtered[filtered["Device"].isin(filter_device)]
        if filter_failure:
            filtered = filtered[filtered["Failure_Type"].isin(filter_failure)]

        st.dataframe(filtered, use_container_width=True, height=380, hide_index=True)

        st.markdown("### Delete a Record")
        if len(filtered) > 0:
            delete_id = st.selectbox("Select Report ID to delete", filtered["Report_ID"].tolist())
            if st.button("Delete Selected Report"):
                st.session_state.reports = st.session_state.reports[
                    st.session_state.reports["Report_ID"] != delete_id
                ]
                st.success(f"Report {delete_id} has been deleted.")
                st.rerun()
        else:
            st.info("No records available to delete.")

        if len(filtered) > 0:
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("Download Filtered Data", csv, "medical_device_reports.csv", "text/csv")

# -------------------------------------------------
# App Entry Point
# -------------------------------------------------
if not st.session_state.logged_in:
    login_page()
else:
    main_app()
