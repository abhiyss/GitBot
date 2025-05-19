import streamlit as st
import sys
import os

sys.path.append("./gitbot")
sys.path.append("./")
#sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def main():

    with open("./streamlit/custom.css") as css:
    #css_path = os.path.join(os.path.dirname(__file__), "..", "custom.css")
    #with open(css_path) as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    # creating a dictionary for name and pages

    pg = st.navigation(
        [
            st.Page("chat_interface.py", title="🚀 GitBot"),
            st.Page("create_agent.py", title="✨ Create new agent"),
            st.Page("existing_agent.py", title="🛠 Existing agent"),
        ]
    )
    pg.run()


if __name__ == "__main__":

    print("Current working directory:", os.getcwd())
    main()
