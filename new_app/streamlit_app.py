import streamlit as st

st.set_page_config(page_title="新しいWebアプリ", page_icon="🛠️", layout="wide")
st.title("新しいWebアプリ")
st.caption("GitHub + Streamlit 開発用ひな型")
st.write("ここから画面と処理を実装できます。")
with st.form("startup_check"):
    name = st.text_input("動作確認用の名前", value="Koji")
    submitted = st.form_submit_button("動作を確認")
if submitted:
    st.success(f"{name.strip() or 'ユーザー'}さん、Streamlitの入力と処理が動作しています。")
