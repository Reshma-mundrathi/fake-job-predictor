import streamlit as st, sys, platform
st.set_page_config(page_title="Env Check")
st.title("✅ Streamlit env check")

st.write({"python": sys.version.split()[0],
          "platform": platform.platform()})
st.success("If you see this, Streamlit is fine.")
