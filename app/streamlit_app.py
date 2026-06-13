"""Streamlit demo. Run:

    uv run streamlit run app/streamlit_app.py

Type a query, pick a method, and see matching clothes with their images.
Search uses text only; the image is just displayed.

If you enable the RAG recommendation, a local LLM (via Ollama) writes a short
explained suggestion. That requires Ollama running and the reviews index built.
"""

import os

import streamlit as st

from fashion_ir.search import Retriever


@st.cache_resource
def get_retriever():
    # Cached so the models load only once, not on every interaction.
    return Retriever(load_cross_encoder=True)


@st.cache_resource
def get_rag(_retriever):
    # Built lazily so the app still works even without the reviews index.
    from fashion_ir.rag import RagRecommender
    return RagRecommender(_retriever)


st.set_page_config(page_title="Fashion Finder", page_icon="👗", layout="wide")
st.title("👗 Fashion Finder — text-based clothing search")
st.caption("Search by occasion, colour, season... images are shown, not indexed.")

retriever = get_retriever()

query = st.text_input("What are you looking for?",
                      placeholder="e.g. blue casual t-shirt for summer")

col1, col2, col3, col4, col5, col6 = st.columns(6)
method = col1.selectbox("Method", ["hybrid", "bm25", "tfidf", "semantic"])
top_k = col2.slider("Results", 3, 20, 9)
expand = col3.checkbox("Query expansion")
occasion = col4.checkbox("Occasion")
do_rerank = col5.checkbox("Rerank")
use_rag = col6.checkbox("Smart pick (RAG)")

if query:
    # Optional RAG recommendation, shown above the grid.
    if use_rag:
        with st.spinner("Asking the local LLM for a recommendation..."):
            try:
                out = get_rag(retriever).recommend(query)
                st.subheader("Recommendation")
                st.write(out["answer"])
                with st.expander("Customer opinions used"):
                    for o in out["opinions"]:
                        st.caption(
                            f"worn for {o['metadata'].get('occasion')} - "
                            f"rating {o['metadata'].get('rating')} - {o['text'][:200]}"
                        )
            except Exception as e:
                st.error(
                    "Could not run RAG. Make sure Ollama is running, the model is "
                    "pulled (ollama pull llama3.2), and the reviews index is built.\n\n"
                    f"{e}"
                )

    # The normal results grid.
    results = retriever.search(
        query, method=method, top_k=top_k, expand=expand,
        occasion=occasion, do_rerank=do_rerank
    )
    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            if os.path.exists(r["image_path"]):
                st.image(r["image_path"], use_container_width=True)
            st.markdown(f"**{r['metadata']['name']}**")
            st.caption(
                f"{r['metadata']['colour']} - {r['metadata']['type']} - "
                f"{r['metadata']['section']} - score {r['score']:.3f}"
            )
