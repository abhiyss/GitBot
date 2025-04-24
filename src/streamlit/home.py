import streamlit as st
import sys

sys.path.append("./gitbot")
sys.path.append("./")


def main():

    with open("./gitbot/streamlit/custom.css") as css:
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
    main()
