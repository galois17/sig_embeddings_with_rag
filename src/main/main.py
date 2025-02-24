import dotenv
import os
import faiss
import numpy as np
import openai
import pandas as pd
import argparse
import streamlit as st
import toml
import json
import requests
import ast
import io
from PIL import Image
import yaml
from string import Template
from jinja2 import Template
import time
import datetime
from functools import wraps

dotenv.load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Rate Limiting Decorator
def rate_limit(period=30): 
    """
    Rate limit calls (default of 30 seconds).
    """
    def decorator(func):
        @wraps(func)  # Preserve original function metadata
        def wrapper(*args, **kwargs):
            now = datetime.datetime.now()
            if "last_api_call" not in st.session_state or \
                st.session_state.last_api_call is None or \
                (now - st.session_state.last_api_call).total_seconds() >= period:
                result = func(*args, **kwargs)
                st.session_state.last_api_call = now
                return result
            else:
                time_remaining = period - (now - st.session_state.last_api_call).total_seconds()
                minutes_remaining = int(time_remaining // 60)
                seconds_remaining = int(time_remaining % 60)
                st.warning(f"Wait {minutes_remaining} minutes {seconds_remaining} seconds before making another request.")
                return None
        return wrapper
    return decorator

def retrieve_similar_captions(captions_df, index, query_embedding, k=3):
    """
    Given a new seaweed image embedding, find k most similar examples in FAISS index.
    """
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
    _, indices = index.search(query_embedding, k)  # Retrieve top-k closest embeddings

    # Extract captions for retrieved indices
    retrieved_captions = [captions_df.iloc[i]["caption"] for i in indices[0]]

    return retrieved_captions


def string_to_dict(input_string):
    """
    Converts a multi-line string to a dictionary.
    """

    result_dict = {}
    lines = input_string.strip().split(
        "\n"
    )

    for line in lines:
        if ":" not in line:
            continue

        key, value = line.split(":", 1)  # Split by the *first* colon only
        key = key.strip()
        value = value.strip()

        # Handle quoted strings
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]  # Remove surrounding double quotes.

        result_dict[key] = value

    return result_dict

def load_prompt(filename, prompt_name):
    """ """
    with open(filename, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts[prompt_name]["prompt"]

def generate_response(prompt_template, query_embedding, retrieved_caption):
    """ """
    template = Template(prompt_template)
    return template.render({'query_embedding':query_embedding, 'retrieved_caption':retrieved_caption})

@rate_limit()
def generate_seaweed_caption(client, captions_df, index, query_embedding):
    """
    Uses FAISS to retrieve similar captions and prompts GPT-4 for the final caption.
    """
    # Retrieve top-3 similar captions
    retrieved_captions = retrieve_similar_captions(captions_df, 
                                                   index, query_embedding, k=3)
    system_prompt_template = load_prompt("prompts.yaml", "marine_prompt")
    system_prompt = generate_response(system_prompt_template, query_embedding, retrieved_captions[0])  

    messages = [
        {"role": "user", "content": system_prompt},
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.9,
        max_tokens=200
    )

    # Extract and print the generated caption
    generated_caption = response.choices[0].message.content
    print("Generated Caption:", generated_caption)
    return generated_caption

def update_css():
    """ """
    # CSS styling for the button
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0099ff;  /* Button background color */
        color: white;              /* Text color */
        padding-top: 20px;         /* Increase padding for height */
        padding-bottom: 20px;      /* Increase padding for height */
        padding-left: 40px;        /* Increase padding for width */
        padding-right: 40px;       /* Increase padding for width */
        font-size: 100px;           /* Increase font size */
        border-radius: 7px;       /* Rounded corners */
        border: none;              /* Remove the default border */
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2); /* Add a subtle shadow */
    }
    div.stButton > button:first-child:hover {
        background-color: #007acc;  /* Change color on hover */
        color: #f0f0f0;              /* Lighter text on hover */
    }
    div.stButton > button:first-child:active {
        position: relative;          /* Slightly move the button on click */
        top: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

def process_embedding(client, captions_df, index, new_embedding):
    print(f"Raw embedding: {new_embedding}")
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}
    
    embedding_str = ",".join([str(x) for x in new_embedding])
    if embedding_str in st.session_state.response_cache:
        st.info("Found response in the cache.")
        caption_as_dict = st.session_state.response_cache[embedding_str]
    else:
        caption  = generate_seaweed_caption(client, captions_df, index, new_embedding)
        if caption:
            st.session_state.call_count = st.session_state.get("call_count", 0) + 1 
            caption_as_dict = string_to_dict(caption)
            st.session_state.response_cache[embedding_str] = caption_as_dict
        else:
            return

    col1, col2, col3 = st.columns(3)
    fucus_per = float(caption_as_dict['Fucus'].replace('%', ''))/100
    asco_per = float(caption_as_dict['Asco'].replace('%', ''))/100
    rest_per = 1.0 - fucus_per - asco_per

    with col1:
        st.subheader("Fucus Coverage", anchor='k1') 
        st.metric("Percentage", caption_as_dict['Fucus'])
        st.progress(fucus_per)

    with col2:
        st.subheader("Asco Coverage")
        st.metric("Percentage", caption_as_dict['Asco'])
        st.progress(asco_per)
    
    with col3:
        st.subheader("Remaining Coverage")
        st.metric("Percentage", f"{rest_per*100:.0f}%")
        st.progress(rest_per)

    st.markdown("---")
    st.subheader(f"Observations: {caption_as_dict['Other']}")
    st.markdown(f"""
                <div style="
                    background-color: #f0f2f6;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 5px solid #007bff;  /* Blue accent line */
                ">
                    <p style="font-size: 1.5em; font-style: italic; color: #333;">
                        {caption_as_dict['Story']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
    )

def main():
    st.set_page_config(page_title="Marine Landscape", layout="wide")  # Set a wider layout
    update_css()

    # Load the saved FAISS index
    config = toml.load("config.toml")
    input_file_path = config["input_file"]
    if not os.path.exists(input_file_path):
        raise ValueError("Supply a FAISS index file path in config.toml")

    index = faiss.read_index(input_file_path)

    captions_df_path = config["captions"]
    if not os.path.exists(captions_df_path):
        raise ValueError("Supply a captions df file path in config.toml")
    
    captions_df = pd.read_csv(captions_df_path)
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    st.title("A Moonlit Marine Tableau")
    st.text("(Siamese network was trained in area of mostly fucus and asco seaweed species...)") 
    user_input = st.text_area(
        "Enter the embedding:",
        placeholder="""[-0.2879038 ,  0.00277467, -0.23789813, -0.16470747, -0.36744052,
        -0.19136755, -0.0723161 ,  0.11992071, -0.00212295, -0.23359874,
         0.2599904 ,  0.00269173,  0.17385477, -0.15069602, -0.01112346,
        -0.26239097,  0.10531765,  0.1856052 , -0.05377749, -0.11307743,
         0.11436549,  0.18217856, -0.24487908,  0.19925837,  0.23600164,
        -0.26388025,  0.0616321 ,  0.10541023, -0.08139204,  0.11356585,
         0.13498148, -0.03214637]""",
        height=200,
        max_chars=4000,
    )

    st.markdown("<div id='my-section-results'></div>", unsafe_allow_html=True) 

    if st.button("Generate analysis"):
        try:
            new_embedding = json.loads(user_input)
        except json.JSONDecodeError as e:
            # Failed parsing as json. Try as a literal
            user_input = user_input.replace("\n", "")
            print(user_input)
            
            new_embedding = ast.literal_eval(user_input)
        
        if len(new_embedding) != 32:
            st.error("The embedding should be a 32 dim feature.")
        else:
            with st.spinner("Thinking..."):
                process_embedding(client, captions_df, index, new_embedding)

                st.components.v1.html('''
                <script>
                    // Time of creation of this script = {now}.
                    function scrollToMySection() {{
                        var element = window.parent.document.getElementById("my-section-{tab_id}");
                        if (element) {{
                            element.scrollIntoView({{ behavior: "smooth", block: "start", inline: "nearest"}});
                        }} else {{
                            setTimeout(scrollToMySection, 200);
                        }}
                    }}
                    scrollToMySection();
                </script>
                '''.format(now=time.time(), tab_id="results"))
                

    st.write("Number of API Calls:", st.session_state.get("call_count", 0))
    if "last_api_call" in st.session_state and st.session_state.last_api_call:
        st.write("Last API call:", st.session_state.last_api_call.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == '__main__':
    main()
