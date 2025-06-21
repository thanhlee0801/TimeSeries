import streamlit as st

try:
    with open("BTL_TimeSeries/results/typhoon_forecast_report.htmll", "r", encoding="utf-8") as f:
        html_from_file = f.read()
    st.markdown(html_from_file, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("Không tìm thấy file 'my_html_content.html'. Đảm bảo nó cùng thư mục với app.py")

st.write("---")

st.write("Các phần còn lại của ứng dụng Streamlit của bạn...")
