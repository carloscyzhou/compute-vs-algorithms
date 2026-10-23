import streamlit as st

st.set_page_config(page_title="What Drives AI Capability?", page_icon="📈", layout="wide")

st.title("What has actually driven AI capability gains?")
st.markdown("More training compute, or better algorithms?")

st.subheader("Audience")
st.markdown(
    "**An Australian Government department** deciding whether a national AI budget should go "
    "to compute (chips, data centres) or to research into better methods."
)

st.caption("Data: Epoch AI, Data on Notable AI Models; Epoch AI, Capabilities & Benchmarking "
           "(retrieved 23 Oct 2026, CC BY 4.0).")
