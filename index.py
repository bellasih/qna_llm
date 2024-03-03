import streamlit as st
from streamlit_chat import message as st_message
from rag import *

model_kwargs = {'device': "cuda"}
encode_kwargs = {'normalize_embeddings': "True"}
embeddings = define_embedding("sentence-transformers/all-mpnet-base-v2", model_kwargs, encode_kwargs)

db = load_faiss_db("faiss_index_all-mpnet-base-v2_cs500_co50_1000", embeddings=embeddings)
model, tokenizer = define_llm("mistralai/Mistral-7B-Instruct-v0.1", True)
llm_chain = create_response_chain(model, tokenizer, 256, temperature=0.1)

# Set up the Streamlit app
st.title("🤖 Chat with your Website")
st.markdown(
    """ 
    ####  🗨️ Chat with your a website 📜  
    """
)

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about this website 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] =  ["Hey ! 👋"]

#container for the chat history and user's text input
response_container, container = st.container(), st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        # Allow the user to enter a query and generate a response
        user_input  = st.text_input(
            "**Talk with your website here**",
            placeholder="Talk with your website here.",
        )
        submit_button = st.form_submit_button(label='Send')

        if user_input:
            with st.spinner(
                "Generating Answer to your Query : `{}` ".format(user_input )
            ):
                answer, relevant_docs = answer_with_rag(user_input, llm_chain, db, k_retrieve=10)

                source = []
                for x in relevant_docs:
                    source.append(x)
                    print(x)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(answer)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            st_message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
            st_message(st.session_state["generated"][i], key=str(i), avatar_style="croodles-neutral")



