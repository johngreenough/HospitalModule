import streamlit as st
import pandas as pd
import numpy as np
import joblib

CLEANED_DATA_PATH = "cleaned.csv"

DEPARTMENTS = ["Emergency", "Surgery", "Critical Care", "Step Down"]

ARRIVAL_RULES = {
    "Emergency": (1.0, 9.0, 4.0),
    "Surgery": (0.0, 2.0, 0.3),
    "Critical Care": (0.0, 2.0, 0.3),
    "Step Down": (0.0, 2.0, 0.3),
}

TRANSFER_RULES = [
    ("Emergency", "Surgery", 0.0, 2.0, 0.5),
    ("Emergency", "Critical Care", 0.0, 2.0, 0.5),
    ("Emergency", "Step Down", 0.0, 2.0, 0.5),
    ("Critical Care", "Surgery", 0.0, 1.0, 0.3),
    ("Critical Care", "Step Down", 0.0, 1.0, 0.3),
    ("Surgery", "Step Down", 0.0, 1.0, 0.3),
    ("Surgery", "Critical Care", 0.0, 1.0, 0.3),
    ("Step Down", "Critical Care", 0.0, 1.0, 0.3),
    ("Step Down", "Surgery", 0.0, 1.0, 0.3),
]

EXIT_RULES = {
    "Emergency": (0.0, 4.0, 2.0),
    "Surgery": (0.0, 2.0, 0.3),
    "Critical Care": (0.0, 2.0, 0.3),
    "Step Down": (0.0, 2.0, 0.3),
}


@st.cache_resource
def load_model():
    model = joblib.load("best_triage_model.pkl")
    return model


@st.cache_data
def load_ktas5_exit_policy(path=CLEANED_DATA_PATH):
    try:
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        if "KTAS_expert" not in df.columns:
            raise ValueError("KTAS_expert column not found in cleaned.csv")

        ktas_series = pd.to_numeric(df["KTAS_expert"], errors="coerce").dropna()
        total = int(len(ktas_series))
        ktas5_count = int((ktas_series == 5).sum())
        rate = float(ktas5_count / total) if total > 0 else 0.0
        return {"rate": rate, "count": ktas5_count, "total": total}
    except Exception:
        return {"rate": 0.0, "count": 0, "total": 0}


def bounded_normal_sample(rng, low, high, mean):
    std = max((high - low) / 6.0, 1e-6)
    return float(np.clip(rng.normal(loc=mean, scale=std), low, high))


def simulate_next_hour(waiting_patients, rng, ktas5_exit_rate):
    patients = {dept: float(waiting_patients[dept]) for dept in DEPARTMENTS}

    for dept in DEPARTMENTS:
        low, high, mean = ARRIVAL_RULES[dept]
        arrivals = bounded_normal_sample(rng, low, high, mean)
        if dept == "Emergency":
            # KTAS 5 patients are discharged home immediately on arrival.
            arrivals *= max(0.0, 1.0 - float(ktas5_exit_rate))
        patients[dept] += arrivals

    inbound_transfers = {dept: 0.0 for dept in DEPARTMENTS}

    for source in DEPARTMENTS:
        desired_transfers = {}
        for src, dst, low, high, mean in TRANSFER_RULES:
            if src == source:
                desired_transfers[dst] = bounded_normal_sample(rng, low, high, mean)

        exit_low, exit_high, exit_mean = EXIT_RULES[source]
        desired_exit = bounded_normal_sample(rng, exit_low, exit_high, exit_mean)
        desired_total_out = desired_exit + sum(desired_transfers.values())

        available = patients[source]
        scale = min(1.0, available / desired_total_out) if desired_total_out > 0 else 1.0

        actual_exit = desired_exit * scale
        actual_transfers = {dst: value * scale for dst, value in desired_transfers.items()}

        patients[source] = max(
            0.0, patients[source] - actual_exit - sum(actual_transfers.values())
        )
        for dst, value in actual_transfers.items():
            inbound_transfers[dst] += value

    for dept in DEPARTMENTS:
        patients[dept] += inbound_transfers[dept]

    return patients


def forecast_staffing(
    free_workers, waiting_patients, simulations, confidence, ktas5_exit_rate
):
    rng = np.random.default_rng()
    sim_results = {dept: [] for dept in DEPARTMENTS}

    for _ in range(simulations):
        end_hour = simulate_next_hour(waiting_patients, rng, ktas5_exit_rate)
        for dept in DEPARTMENTS:
            sim_results[dept].append(end_hour[dept])

    sim_df = pd.DataFrame(sim_results)
    mean_patients = sim_df.mean()
    confidence_patients = sim_df.quantile(confidence)

    result_rows = []
    for dept in DEPARTMENTS:
        required_workers = int(np.ceil(confidence_patients[dept]))
        extra_staff = max(required_workers - int(free_workers[dept]), 0)
        current_shortfall = max(int(waiting_patients[dept]) - int(free_workers[dept]), 0)
        result_rows.append(
            {
                "Department": dept,
                "Free Workers": int(free_workers[dept]),
                "Current Waiting Patients": int(waiting_patients[dept]),
                "Current Shortfall": current_shortfall,
                "Forecast Patients (Mean)": round(float(mean_patients[dept]), 2),
                f"Forecast Patients (P{int(confidence * 100)})": int(
                    np.ceil(confidence_patients[dept])
                ),
                "Recommended Extra Staff": extra_staff,
            }
        )

    return pd.DataFrame(result_rows)


model = load_model()
ktas5_policy = load_ktas5_exit_policy()
KTAS5_EXIT_RATE = ktas5_policy["rate"]

st.title("Hospital Triage Decision Support System")

triage_tab, staffing_tab = st.tabs(["Triage", "Staffing"])

with triage_tab:
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.number_input("Age", 0, 120, 50)
    injury = st.selectbox("Injury", ["No", "Yes"])
    pain = st.selectbox("Pain Present?", ["No", "Yes"])
    mental = st.selectbox(
        "Mental State",
        ["Alert", "Verbal Response", "Pain Response", "Unresponsive"],
    )

    sbp = st.number_input("Systolic BP", 60, 250, 120)
    dbp = st.number_input("Diastolic BP", 30, 150, 80)
    hr = st.number_input("Heart Rate", 30, 200, 80)
    rr = st.number_input("Respiratory Rate", 5, 60, 18)
    temp = st.number_input("Temperature (C)", 30.0, 45.0, 36.5)
    saturation = st.number_input("Oxygen Saturation (%)", 70.0, 100.0, 98.0)
    nrs_pain = st.number_input("Pain Score (0-10)", 0, 10, 3)
    ktas_rn = st.number_input("KTAS Score by Nurse", 1, 5, 3)

    mental_order = ["Alert", "Verbal Response", "Pain Response", "Unresponsive"]
    mental_ord = mental_order.index(mental) + 1

    input_data = {
        "Age": age,
        "NRS_pain": nrs_pain,
        "SBP": sbp,
        "DBP": dbp,
        "HR": hr,
        "RR": rr,
        "BT": temp,
        "Saturation": saturation,
        "KTAS_RN": ktas_rn,
        "Sex_Male": int(sex == "Male"),
        "Injury_Yes": int(injury == "Yes"),
        "Pain_Yes": int(pain == "Yes"),
        "Mental_ord": mental_ord,
    }

    if st.button("Predict Triage Level"):
        input_df = pd.DataFrame([input_data])
        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)
            decoded_pred = pred + 1

            st.success(f"Predicted Triage Level: **KTAS {decoded_pred}**")
            if decoded_pred == 5:
                st.warning(
                    "Policy triggered: KTAS 5 patients are discharged home immediately and removed from the ED system."
                )

            proba_df = pd.DataFrame(
                proba, columns=[f"KTAS {i + 1}" for i in range(proba.shape[1])]
            )
            highlight = [
                "background-color: yellow" if i == pred else ""
                for i in range(proba.shape[1])
            ]
            st.dataframe(proba_df.style.apply(lambda _: highlight, axis=1))

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write(input_df)

with staffing_tab:
    st.subheader("Staffing Forecast (Next Hour)")
    st.caption(
        "Forecast assumes 1 patient requires 1 worker. Recommendations are based on a simulation percentile."
    )
    if ktas5_policy["total"] > 0:
        st.caption(
            f"KTAS 5 immediate-home policy active for Emergency arrivals using cleaned.csv frequency: "
            f"{ktas5_policy['count']}/{ktas5_policy['total']} ({KTAS5_EXIT_RATE * 100:.2f}%)."
        )
    else:
        st.caption(
            "KTAS 5 immediate-home policy active, but cleaned.csv KTAS_expert frequency could not be loaded. "
            "Fallback rate: 0.00%."
        )

    worker_cols = st.columns(4)
    waiting_cols = st.columns(4)

    free_workers = {}
    waiting_patients = {}

    for idx, dept in enumerate(DEPARTMENTS):
        free_workers[dept] = worker_cols[idx].number_input(
            f"{dept} free workers", min_value=0, value=2, step=1, key=f"free_{dept}"
        )
        waiting_patients[dept] = waiting_cols[idx].number_input(
            f"{dept} waiting patients", min_value=0, value=0, step=1, key=f"wait_{dept}"
        )

    settings_col1, settings_col2 = st.columns(2)
    simulations = settings_col1.number_input(
        "Simulation runs", min_value=500, max_value=20000, value=5000, step=500
    )
    confidence = settings_col2.slider(
        "Coverage percentile", min_value=0.5, max_value=0.99, value=0.9, step=0.01
    )

    if st.button("Forecast Extra Staff Needed"):
        result_df = forecast_staffing(
            free_workers=free_workers,
            waiting_patients=waiting_patients,
            simulations=int(simulations),
            confidence=float(confidence),
            ktas5_exit_rate=KTAS5_EXIT_RATE,
        )
        st.dataframe(result_df, use_container_width=True)

        total_extra = int(result_df["Recommended Extra Staff"].sum())
        st.metric("Total Extra Staff To Call", total_extra)
