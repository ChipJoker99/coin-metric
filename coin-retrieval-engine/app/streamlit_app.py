"""
Streamlit UI for the Coin Retrieval Engine.

Allows the user to upload a coin image and visualize the top-K most similar
coins retrieved from the pre-built embedding index.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from inference.predict import CoinPredictor  # noqa: E402

INDEX_PATH = ROOT / "data" / "embeddings" / "index.pkl"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"


def _latest_checkpoint() -> Path | None:
    if not CHECKPOINTS_DIR.exists():
        return None
    checkpoints = sorted(CHECKPOINTS_DIR.glob("*.pt"))
    return checkpoints[-1] if checkpoints else None


@st.cache_resource(show_spinner="Caricamento modello e indice...")
def _load_predictor() -> CoinPredictor | None:
    if not INDEX_PATH.exists():
        return None
    checkpoint = _latest_checkpoint()
    if checkpoint:
        return CoinPredictor.from_checkpoint(
            checkpoint_path=checkpoint,
            index_path=INDEX_PATH,
            device="cpu",
        )
    return CoinPredictor.from_paths(index_path=INDEX_PATH, device="cpu")


st.set_page_config(page_title="Coin Retrieval Demo", layout="wide")
st.title("🪙 Coin Retrieval Demo")
st.write("Carica un'immagine di una moneta per trovare le più simili nel dataset.")

predictor = _load_predictor()

if predictor is None:
    st.error(
        "Indice non trovato. "
        f"Esegui prima `python scripts/build_index.py` per generare `{INDEX_PATH}`."
    )
    st.stop()

checkpoint = _latest_checkpoint()
if checkpoint:
    st.caption(f"Modello caricato da: `{checkpoint.name}`")
else:
    st.caption("Nessun checkpoint trovato — uso pesi casuali (esegui il training prima).")

st.divider()

col_upload, col_results = st.columns([1, 2])

with col_upload:
    st.subheader("Immagine query")
    uploaded = st.file_uploader(
        "Seleziona un'immagine",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    top_k = st.slider("Numero di risultati (top-k)", min_value=1, max_value=10, value=5)

    if uploaded:
        query_image = Image.open(uploaded).convert("RGB")
        st.image(query_image, caption="Query", use_container_width=True)
        run_button = st.button("🔍 Trova monete simili", type="primary")
    else:
        run_button = False

with col_results:
    st.subheader("Monete simili")

    if not uploaded:
        st.info("Carica un'immagine per iniziare.")
    elif run_button:
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                query_image.save(tmp_path, format="JPEG")

            with st.spinner("Ricerca in corso..."):
                results = predictor.predict(tmp_path, top_k=top_k)

            tmp_path.unlink(missing_ok=True)

            if not results:
                st.warning("Nessun risultato trovato.")
            else:
                cols = st.columns(min(len(results), 5))
                for i, result in enumerate(results):
                    col = cols[i % len(cols)]
                    result_path = Path(result.get("path", ""))
                    label = result.get("label", "—")
                    score = result.get("score", 0.0)

                    with col:
                        if result_path.exists():
                            st.image(str(result_path), use_container_width=True)
                        else:
                            st.markdown("🪙 *(immagine non disponibile)*")
                        st.caption(f"**{label}**  \nScore: `{score:.4f}`")

        except Exception as exc:
            st.error(f"Errore durante la predizione: {exc}")
