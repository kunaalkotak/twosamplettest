import streamlit as st
import numpy as np
from statistics import stdev
from scipy.stats import t
from scipy import stats

# -------------------------------
# Two-sample t-test function
# -------------------------------
def two_sample(a, b, alternative):
    xbar1 = np.mean(a)
    xbar2 = np.mean(b)
    sd1 = stdev(a)
    sd2 = stdev(b)
    n1 = len(a)
    n2 = len(b)

    alpha = 0.05 / 2
    df = n1 + n2 - 2

    se = np.sqrt((sd1**2)/n1 + (sd2**2)/n2)
    tcal = (xbar1 - xbar2) / se

    t_table_pos = t.ppf(1 - alpha, df)
    t_table_neg = t.ppf(alpha, df)

    if alternative == "two-sided":
        p_value = 2 * (1 - t.cdf(abs(tcal), df))
    elif alternative == "left":
        p_value = t.cdf(tcal, df)
    else:  # right
        p_value = 1 - t.cdf(tcal, df)

    scipy_result = stats.ttest_ind(
        a, b, alternative=alternative, equal_var=False
    )

    return {
        "mean1": xbar1,
        "mean2": xbar2,
        "t_calculated": tcal,
        "t_table_positive": t_table_pos,
        "t_table_negative": t_table_neg,
        "p_value": p_value,
        "scipy_result": scipy_result
    }

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Two Sample t-Test", layout="centered")

st.title("📊 Two Sample t-Test (Streamlit App)")
st.write("Enter two independent samples and choose the alternative hypothesis.")

sample1 = st.text_input(
    "Sample 1 (comma-separated values)",
    "10,12,14,15,16"
)

sample2 = st.text_input(
    "Sample 2 (comma-separated values)",
    "8,9,11,13,14"
)

alternative = st.selectbox(
    "Alternative Hypothesis",
    ["two-sided", "left", "right"]
)

if st.button("Run t-Test"):
    try:
        a = np.array([float(x) for x in sample1.split(",")])
        b = np.array([float(x) for x in sample2.split(",")])

        if len(a) < 2 or len(b) < 2:
            st.error("Each sample must have at least two observations.")
        else:
            result = two_sample(a, b, alternative)

            st.subheader("📌 Test Results")
            st.write(f"Mean of Sample 1: **{result['mean1']:.4f}**")
            st.write(f"Mean of Sample 2: **{result['mean2']:.4f}**")
            st.write(f"Calculated t-value: **{result['t_calculated']:.4f}**")
            st.write(f"t-table (positive): **{result['t_table_positive']:.4f}**")
            st.write(f"t-table (negative): **{result['t_table_negative']:.4f}**")
            st.write(f"P-value: **{result['p_value']:.6f}**")

            st.subheader("🧪 SciPy Verification")
            st.write(result["scipy_result"])

            if result["p_value"] < 0.05:
                st.success("Reject the Null Hypothesis ✅")
            else:
                st.info("Fail to Reject the Null Hypothesis ❌")

    except Exception as e:
        st.error(f"Error: {e}")