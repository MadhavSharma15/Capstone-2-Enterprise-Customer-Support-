import uuid
import requests
import streamlit as st

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="EnterpriseAssist | Customer Support AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/query"

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f5f8fc;
}

[data-testid="stHeader"] {
    background: rgba(245,248,252,0.9);
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

[data-testid="stSidebar"] {
    background: #102a43;
}

[data-testid="stSidebar"] * {
    color: #f2f7fc;
}

.brand {
    font-size: 25px;
    font-weight: 700;
    color: white;
    padding: 10px 0;
}

.brand span {
    color: #62e6c7;
}

.sidebar-note {
    color: #bdcfe0;
    font-size: 13px;
    line-height: 1.7;
}

.hero {
    background: linear-gradient(120deg,#123b56,#087f8c);
    padding: 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 24px;
}

.hero-tag {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 30px;
    background: rgba(255,255,255,0.15);
    margin-bottom: 12px;
    font-size: 12px;
}

.hero h1 {
    color: white;
}

.info-card {
    background: white;
    padding: 17px;
    border-radius: 15px;
    border: 1px solid #e2eaf2;
    min-height: 120px;
}

.info-card h4 {
    color: #123b56;
}

.info-card p {
    color: #60758a;
    font-size: 13px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #16324f;
    margin: 20px 0;
}

.disclaimer {
    background: #fff8e8;
    color: #775a20;
    padding: 13px;
    border-radius: 12px;
    border: 1px solid #f3e3b8;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #8192a5;
    font-size: 12px;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="brand">🏢 Enterprise<span>Assist</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sidebar-note">Your AI-powered Enterprise Customer Support Assistant.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    if st.button("＋ Start New Chat", use_container_width=True):

        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("### Capabilities")

    st.markdown(
        """
<div class="sidebar-note">

◈ Enterprise Knowledge Retrieval

◈ IT Support Assistance

◈ HR Policy Guidance

◈ Ticket Resolution Support

◈ Account & Access Help

◈ Context Aware Responses

</div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
<div class="sidebar-note">

<b>Security Reminder</b><br>

Do not enter passwords, confidential company information,
or sensitive employee details.

</div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "EnterpriseAssist • Agentic RAG Capstone Project"
    )

# ---------------------------------------------------
# HERO
# ---------------------------------------------------

st.markdown(
    """
<div class="hero">

<div class="hero-tag">

💼 AI POWERED ENTERPRISE SUPPORT

</div>

<h1>
Enterprise Support,<br>
Powered by Agentic AI
</h1>

<p>

Get support for Password Resets, VPN Access,
Software Installation, Account Issues,
HR Policies, Leave Requests and more.

</p>

</div>
""",
    unsafe_allow_html=True
)

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
<div class="info-card">
<h4>🔍 Knowledge Retrieval</h4>
<p>
Retrieve information from policies, FAQs,
and support documentation.
</p>
</div>
""", unsafe_allow_html=True)

with col2:

    st.markdown("""
<div class="info-card">
<h4>🤖 Intelligent Support</h4>
<p>
Receive context-aware enterprise support responses.
</p>
</div>
""", unsafe_allow_html=True)

with col3:

    st.markdown("""
<div class="info-card">
<h4>📋 Policy References</h4>
<p>
Review policy guidance and historical ticket resolutions.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SUGGESTED QUESTIONS
# ---------------------------------------------------

if not st.session_state.messages:

    st.markdown(
        '<div class="section-title">How can we assist you today?</div>',
        unsafe_allow_html=True
    )

    suggestions = [

        "How do I reset my password?",

        "How can I access VPN?",

        "How do I request a laptop?",

        "What is the leave policy?"

    ]

    left, right = st.columns(2)

    for i, question in enumerate(suggestions):

        with (left if i % 2 == 0 else right):

            if st.button(
                question,
                key=f"suggestion_{i}",
                use_container_width=True
            ):

                st.session_state.pending_question = question
                st.rerun()

# ---------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">Conversation</div>',
    unsafe_allow_html=True
)

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

question = st.chat_input(
    "Ask an enterprise support question..."
)

if question:
    st.session_state.pending_question = question

# ---------------------------------------------------
# PROCESS QUERY
# ---------------------------------------------------

if st.session_state.pending_question:

    question = st.session_state.pending_question
    st.session_state.pending_question = None

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing your request and retrieving enterprise knowledge..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": question,
                        "thread_id": st.session_state.thread_id
                    },
                    timeout=180
                )

                response.raise_for_status()

                data = response.json()

                answer = data.get(
                    "answer",
                    "No answer returned."
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                error_message = f"Error: {str(e)}"

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown(
"""
<div class="disclaimer">

<b>Enterprise Support Disclaimer:</b>

This assistant provides information from enterprise
knowledge bases, support policies and historical
ticket data.

Responses are intended for informational purposes
and should not replace official IT, HR, Security
or Management decisions.

</div>

<div class="footer">

EnterpriseAssist • Customer Support Assistant •
Powered by Agentic RAG and LangGraph

</div>
""",
unsafe_allow_html=True
)