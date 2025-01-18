import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)

with st.sidebar:
    st.link_button("Visit repository", "https://github.com/Kosto1221/fullstack-gpt")

st.markdown(
    """
    # Hello!

    Welcome to my FullstackGPT portfolio!

    Here are the apps I made:

    - [x] [DocumentGPT](/DocumentGPT)
    - [ ] [PrivateGPT](/PrivateGPT)
    - [x] [QuizGPT](/QuizGPT)
    - [ ] [SiteGPT](/SiteGPT)
    - [ ] [MeetingGPT](/MeetingGPT)
    - [ ] [InvestorGPT](/InvestorGPT)


    """
)