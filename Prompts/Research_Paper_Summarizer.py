from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

st.header("Research Tool")

chat_model = ChatOpenAI(model='gpt-4.1-mini')

paper_input = st.selectbox("Select Research Paper Name", ["Make your selection","Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers","GPT-3: Language Models are Few-Shot Learners","Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Make your selection","Beginner-Friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explanation Length",["Make your selection","Short (1 - 2 paragraphs)","Medium (3 - 5 paragraphs)","Long (detailed explanation)"])


# Template
# template = PromptTemplate(
#     template = """
#     Please summarize the research paper titled "{paper_input}" with the following specifications:
#     Explanation Style: {style_input}
#     Explanation Length: {length_input}
#     1. Mathematical Details:
#         - Include relevant mathematical equations if present in the paper.
#         - Explain the mathematical concepts using simple, intutive code snippets where applicable.
#     2. Analogies:
#         - Use relatable analogies to simplify complex ideas.
#     If certain information is not available in the paper, respond with: "Insufficient information system available" instead of guessing.
#     Ensure the summary is clear, accurate, and aligned with the provided style and length.
#     """,
#     input_variables=['paper_input','style_input','length_input']
# )

template = load_prompt('template.json')

# Fill the Placeholders
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})


if st.button("Summarize"):
    result = chat_model.invoke(prompt)
    st.write(result.content)