from __future__ import annotations

import os

# ============================================================
# HUGGING FACE DOWNLOAD SETTINGS
# ============================================================
# WAJIB berada sebelum import huggingface_hub dan transformers.

# Hindari error transfer Xet pada Streamlit Community Cloud.
# Download checkpoint dilakukan melalui HTTP biasa.
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Tambahkan batas waktu untuk file model berukuran besar.
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"

# Gunakan cache baru agar cache parsial lama tidak dipakai kembali.
os.environ["HF_HUB_CACHE"] = (
    "/tmp/cognitive_distortion_hf_cache_v3"
)


import gc
import json
import re
import time

from pathlib import Path
from typing import Any

import streamlit as st
import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cognitive Distortion Detector",
    page_icon="🧠",
    layout="wide"
)

# Batasi thread CPU agar lebih aman pada hosting gratis.
torch.set_num_threads(
    2
)


# ============================================================
# LOCAL PATH CONFIGURATION
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

BINARY_CONFIG_PATH = (
    BASE_DIR
    / "models"
    / "binary"
    / "binary_config.json"
)

MULTICLASS_CONFIG_PATH = (
    BASE_DIR
    / "models"
    / "multiclass"
    / "multiclass_config.json"
)

MULTICLASS_LABELS_PATH = (
    BASE_DIR
    / "models"
    / "multiclass"
    / "multiclass_labels.json"
)


# ============================================================
# HUGGING FACE REPOSITORY CONFIGURATION
# ============================================================

HF_REPO_ID = (
    "jjunion/"
    "cognitive-distortion-indobert-v2"
)

HF_CACHE_DIR = Path(
    os.environ[
        "HF_HUB_CACHE"
    ]
)

HF_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEFAULT_SEEDS = [
    42,
    123,
    2025
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# JSON CONFIGURATION HELPERS
# ============================================================

def read_json(
    file_path: Path
) -> Any:

    if not file_path.exists():

        raise FileNotFoundError(
            f"File konfigurasi tidak ditemukan: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


def validate_configuration(
    binary_config: dict,
    multiclass_config: dict,
    multiclass_labels: list[str]
) -> None:

    required_binary_keys = {
        "model_name",
        "max_len",
        "num_classes",
        "threshold",
        "labels"
    }

    required_multiclass_keys = {
        "model_name",
        "max_len",
        "num_classes"
    }

    missing_binary_keys = (
        required_binary_keys
        - set(
            binary_config
        )
    )

    missing_multiclass_keys = (
        required_multiclass_keys
        - set(
            multiclass_config
        )
    )

    if missing_binary_keys:

        raise ValueError(
            "Key binary_config.json belum lengkap: "
            + ", ".join(
                sorted(
                    missing_binary_keys
                )
            )
        )

    if missing_multiclass_keys:

        raise ValueError(
            "Key multiclass_config.json belum lengkap: "
            + ", ".join(
                sorted(
                    missing_multiclass_keys
                )
            )
        )

    if int(
        binary_config[
            "num_classes"
        ]
    ) != 2:

        raise ValueError(
            "num_classes binary harus bernilai 2."
        )

    if len(
        binary_config[
            "labels"
        ]
    ) != 2:

        raise ValueError(
            "Jumlah labels binary harus tepat 2."
        )

    if (
        int(
            multiclass_config[
                "num_classes"
            ]
        )
        != len(
            multiclass_labels
        )
    ):

        raise ValueError(
            "num_classes multiclass tidak sama dengan "
            "jumlah label pada multiclass_labels.json."
        )


@st.cache_data(
    show_spinner=False
)
def load_local_configuration():

    binary_config = read_json(
        BINARY_CONFIG_PATH
    )

    multiclass_config = read_json(
        MULTICLASS_CONFIG_PATH
    )

    multiclass_labels = read_json(
        MULTICLASS_LABELS_PATH
    )

    validate_configuration(
        binary_config=binary_config,
        multiclass_config=multiclass_config,
        multiclass_labels=multiclass_labels
    )

    return (
        binary_config,
        multiclass_config,
        multiclass_labels
    )


def get_seeds(
    config: dict
) -> list[int]:

    seeds = config.get(
        "seeds",
        DEFAULT_SEEDS
    )

    if not seeds:

        raise ValueError(
            "Daftar seed tidak boleh kosong."
        )

    return [
        int(
            seed
        )
        for seed in seeds
    ]


def build_checkpoint_paths(
    stage: str,
    seeds: list[int]
) -> list[str]:

    return [
        (
            f"{stage}/"
            f"{stage}_model_seed{seed}.pt"
        )
        for seed in seeds
    ]


def get_hf_token() -> str | None:

    """
    Repository publik tidak memerlukan token.

    Apabila repository diubah menjadi private,
    simpan HF_TOKEN melalui Streamlit Secrets.
    """

    try:

        return st.secrets.get(
            "HF_TOKEN",
            None
        )

    except Exception:

        return None


# ============================================================
# HUGGING FACE DOWNLOAD HELPERS
# ============================================================

def clear_incomplete_downloads() -> None:

    """
    Hapus file download parsial.

    Checkpoint yang sudah selesai tetap disimpan
    agar dapat digunakan kembali dari cache.
    """

    if not HF_CACHE_DIR.exists():

        return

    for incomplete_file in HF_CACHE_DIR.rglob(
        "*.incomplete"
    ):

        try:

            incomplete_file.unlink()

        except FileNotFoundError:

            pass


def is_retryable_download_error(
    error: Exception
) -> bool:

    error_message = str(
        error
    )

    retryable_markers = (
        "File size mismatch",
        "Consistency check failed",
        "downloaded",
        "incomplete",
        "Timeout",
        "timed out"
    )

    return any(
        marker in error_message
        for marker in retryable_markers
    )


def download_checkpoint(
    filename: str
) -> str:

    """
    Download checkpoint melalui HTTP biasa.

    Percobaan pertama menggunakan cache.
    Jika file parsial atau ukurannya tidak sesuai,
    file parsial dihapus dan download diulang sekali.
    """

    download_arguments = {

        "repo_id":
            HF_REPO_ID,

        "filename":
            filename,

        "repo_type":
            "model",

        "token":
            get_hf_token(),

        "cache_dir":
            str(
                HF_CACHE_DIR
            )
    }

    try:

        return hf_hub_download(
            **download_arguments
        )

    except Exception as error:

        if not is_retryable_download_error(
            error
        ):

            raise

        clear_incomplete_downloads()

        time.sleep(
            1
        )

        return hf_hub_download(
            **download_arguments,
            force_download=True
        )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    text: str,
    cleaning_config: dict | None = None
) -> str:

    """
    Cleaning sama seperti preprocessing notebook.

    Simbol $ dihapus.
    Isi kalimat tetap dipertahankan.
    """

    cleaning_config = (
        cleaning_config
        or {}
    )

    text = str(
        text
    )

    if cleaning_config.get(
        "remove_dollar_symbol",
        True
    ):

        text = re.sub(
            r"\$+",
            " ",
            text
        )

    if cleaning_config.get(
        "normalize_whitespace",
        True
    ):

        text = re.sub(
            r"\s+",
            " ",
            text
        )

    return text.strip()


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class BertClassifier(
    nn.Module
):

    """
    Arsitektur klasifikasi:

    IndoBERT Base P2
    ↓
    CLS Representation
    ↓
    Linear 768 → 512
    ↓
    LayerNorm
    ↓
    GELU
    ↓
    Multi-Sample Dropout
    ↓
    Linear Output
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int
    ):

        super().__init__()

        config = AutoConfig.from_pretrained(
            model_name
        )

        # Backbone dibuat dari konfigurasi.
        # Bobot lengkap akan dimuat dari checkpoint hasil fine-tuning.
        self.bert = AutoModel.from_config(
            config
        )

        hidden_size = (
            self.bert
            .config
            .hidden_size
        )

        self.fc1 = nn.Linear(
            hidden_size,
            512
        )

        self.norm1 = nn.LayerNorm(
            512
        )

        self.gelu = nn.GELU()

        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(
                    0.1
                ),

                nn.Dropout(
                    0.2
                ),

                nn.Dropout(
                    0.3
                ),

                nn.Dropout(
                    0.4
                ),

                nn.Dropout(
                    0.5
                )
            ]
        )

        self.fc2 = nn.Linear(
            512,
            num_classes
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = (
            outputs
            .last_hidden_state[
                :,
                0
            ]
        )

        x = self.fc1(
            cls_output
        )

        x = self.norm1(
            x
        )

        x = self.gelu(
            x
        )

        # Saat model.eval(), seluruh dropout nonaktif.
        # Rata-rata lima cabang sama dengan satu kali fc2(x).
        logits = self.fc2(
            x
        )

        return logits


# ============================================================
# TOKENIZER
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_tokenizer(
    model_name: str
):

    return AutoTokenizer.from_pretrained(
        model_name
    )


def tokenize_text(
    text: str,
    tokenizer,
    max_len: int
) -> dict[str, torch.Tensor]:

    encoding = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    return {

        "input_ids":
            encoding[
                "input_ids"
            ].to(
                DEVICE
            ),

        "attention_mask":
            encoding[
                "attention_mask"
            ].to(
                DEVICE
            )
    }


# ============================================================
# MEMORY HELPERS
# ============================================================

def release_memory() -> None:

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


def load_state_dict_low_memory(
    checkpoint_path: str
):

    """
    mmap=True membantu menekan lonjakan RAM
    saat checkpoint besar dibaca dari disk.
    """

    try:

        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True
        )

    except (
        TypeError,
        RuntimeError
    ):

        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )


def load_weights_into_model(
    model: BertClassifier,
    state_dict: dict
) -> None:

    """
    assign=True membantu mengurangi penggunaan memori
    pada versi PyTorch yang mendukungnya.
    """

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
            assign=True
        )

    except TypeError:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    model.to(
        DEVICE
    )


# ============================================================
# SEQUENTIAL ENSEMBLE INFERENCE
# ============================================================

def predict_stage_sequentially(
    text: str,
    tokenizer,
    model_name: str,
    max_len: int,
    num_classes: int,
    checkpoint_files: list[str],
    stage_label: str
) -> torch.Tensor:

    """
    Hanya satu objek IndoBERT berada di RAM.

    Bobot seed 42, 123, dan 2025 dimuat
    secara bergantian ke objek model yang sama.
    """

    tokens = tokenize_text(
        text=text,
        tokenizer=tokenizer,
        max_len=max_len
    )

    probability_sum = torch.zeros(
        num_classes,
        dtype=torch.float32
    )

    total_models = len(
        checkpoint_files
    )

    if total_models == 0:

        raise ValueError(
            "Checkpoint ensemble tidak tersedia."
        )

    progress_bar = st.progress(
        0,
        text=f"Menyiapkan {stage_label}..."
    )

    model = BertClassifier(
        model_name=model_name,
        num_classes=num_classes
    )

    try:

        for index, checkpoint_file in enumerate(
            checkpoint_files,
            start=1
        ):

            progress_bar.progress(
                int(
                    (
                        index - 1
                    )
                    / total_models
                    * 100
                ),
                text=(
                    f"{stage_label}: "
                    f"memproses model {index}/{total_models}"
                )
            )

            checkpoint_path = download_checkpoint(
                checkpoint_file
            )

            state_dict = load_state_dict_low_memory(
                checkpoint_path
            )

            load_weights_into_model(
                model=model,
                state_dict=state_dict
            )

            del state_dict

            release_memory()

            model.eval()

            with torch.inference_mode():

                logits = model(
                    input_ids=tokens[
                        "input_ids"
                    ],
                    attention_mask=tokens[
                        "attention_mask"
                    ]
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )[
                    0
                ].detach().cpu()

            probability_sum += (
                probabilities
            )

            del logits
            del probabilities

            if DEVICE.type == "cuda":

                model.to(
                    "cpu"
                )

            release_memory()

        progress_bar.progress(
            100,
            text=f"{stage_label} selesai."
        )

        return (
            probability_sum
            / total_models
        )

    finally:

        progress_bar.empty()

        del model
        del tokens

        release_memory()


# ============================================================
# BINARY PREDICTION
# ============================================================

def predict_binary(
    text: str,
    tokenizer,
    binary_config: dict
) -> dict:

    seeds = get_seeds(
        binary_config
    )

    checkpoint_files = build_checkpoint_paths(
        stage="binary",
        seeds=seeds
    )

    probabilities = predict_stage_sequentially(
        text=text,
        tokenizer=tokenizer,
        model_name=binary_config[
            "model_name"
        ],
        max_len=int(
            binary_config[
                "max_len"
            ]
        ),
        num_classes=int(
            binary_config[
                "num_classes"
            ]
        ),
        checkpoint_files=checkpoint_files,
        stage_label="Deteksi binary"
    )

    no_distortion_probability = float(
        probabilities[
            0
        ]
    )

    distortion_probability = float(
        probabilities[
            1
        ]
    )

    threshold = float(
        binary_config.get(
            "threshold",
            0.5
        )
    )

    predicted_index = int(
        distortion_probability
        >= threshold
    )

    return {

        "predicted_index":
            predicted_index,

        "predicted_label":
            binary_config[
                "labels"
            ][
                predicted_index
            ],

        "no_distortion_probability":
            no_distortion_probability,

        "distortion_probability":
            distortion_probability
    }


# ============================================================
# MULTICLASS PREDICTION
# ============================================================

def predict_multiclass(
    text: str,
    tokenizer,
    multiclass_config: dict,
    multiclass_labels: list[str]
) -> dict:

    seeds = get_seeds(
        multiclass_config
    )

    checkpoint_files = build_checkpoint_paths(
        stage="multiclass",
        seeds=seeds
    )

    probabilities = predict_stage_sequentially(
        text=text,
        tokenizer=tokenizer,
        model_name=multiclass_config[
            "model_name"
        ],
        max_len=int(
            multiclass_config[
                "max_len"
            ]
        ),
        num_classes=int(
            multiclass_config[
                "num_classes"
            ]
        ),
        checkpoint_files=checkpoint_files,
        stage_label="Klasifikasi multiclass"
    )

    predicted_index = int(
        torch.argmax(
            probabilities
        )
    )

    ranked_indices = torch.argsort(
        probabilities,
        descending=True
    ).tolist()

    ranked_results = [

        {

            "Jenis Distorsi":
                multiclass_labels[
                    class_index
                ],

            "Probabilitas":
                (
                    f"{float(probabilities[class_index]) * 100:.2f}%"
                )
        }

        for class_index in ranked_indices
    ]

    return {

        "predicted_index":
            predicted_index,

        "predicted_label":
            multiclass_labels[
                predicted_index
            ],

        "confidence":
            float(
                probabilities[
                    predicted_index
                ]
            ),

        "ranked_results":
            ranked_results
    }


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🧠 Cognitive Distortion Detector"
)

st.write(
    """
    Aplikasi ini menganalisis teks berbahasa Indonesia menggunakan
    IndoBERT Base P2 dengan sistem klasifikasi dua tahap.
    Tahap pertama mendeteksi keberadaan cognitive distortion.
    Jika distorsi terdeteksi, tahap kedua menentukan jenisnya.
    """
)

st.info(
    """
    Hasil aplikasi merupakan prediksi model machine learning,
    bukan diagnosis psikologis atau medis.
    """
)


with st.sidebar:

    st.header(
        "Informasi Model"
    )

    st.write(
        "Model dasar: IndoBERT Base P2"
    )

    st.write(
        "Binary ensemble: 3 seed"
    )

    st.write(
        "Multiclass ensemble: 3 seed"
    )

    st.write(
        "Inference: sequential loading"
    )

    st.write(
        "Checkpoint: Hugging Face Hub"
    )

    st.write(
        f"Device: `{DEVICE}`"
    )

    st.caption(
        """
        Satu model dimuat pada satu waktu untuk menghemat RAM.
        Simbol anotasi `$` dibersihkan tanpa menghapus isi kalimat.
        """
    )


try:

    (
        binary_config,
        multiclass_config,
        multiclass_labels
    ) = load_local_configuration()

    tokenizer = load_tokenizer(
        binary_config[
            "model_name"
        ]
    )

except Exception as error:

    st.error(
        "Konfigurasi awal atau tokenizer gagal dimuat."
    )

    st.exception(
        error
    )

    st.stop()


user_text = st.text_area(
    "Masukkan kalimat yang ingin dianalisis:",
    placeholder=(
        "Contoh: Saya merasa selalu gagal "
        "dan tidak akan pernah berhasil."
    ),
    height=150
)

analyze_button = st.button(
    "Analisis Teks",
    type="primary",
    use_container_width=True
)


if analyze_button:

    cleaned_text = clean_text(
        text=user_text,
        cleaning_config=binary_config.get(
            "text_cleaning",
            {}
        )
    )

    if len(
        cleaned_text
    ) <= 5:

        st.warning(
            "Masukkan kalimat yang lebih lengkap."
        )

        st.stop()

    with st.expander(
        "Teks setelah cleaning"
    ):

        st.write(
            cleaned_text
        )

    try:

        binary_result = predict_binary(
            text=cleaned_text,
            tokenizer=tokenizer,
            binary_config=binary_config
        )

    except Exception as error:

        st.error(
            "Model binary gagal dijalankan."
        )

        st.exception(
            error
        )

        st.stop()

    st.divider()

    st.subheader(
        "Tahap 1 — Deteksi Distorsi"
    )

    column_1, column_2 = st.columns(
        2
    )

    with column_1:

        st.metric(
            "No Distortion",
            (
                f"{binary_result['no_distortion_probability'] * 100:.2f}%"
            )
        )

    with column_2:

        st.metric(
            "Distortion",
            (
                f"{binary_result['distortion_probability'] * 100:.2f}%"
            )
        )

    if binary_result[
        "predicted_index"
    ] == 0:

        st.success(
            "Tidak terdeteksi cognitive distortion."
        )

        st.stop()

    st.warning(
        "Terdeteksi indikasi cognitive distortion."
    )

    try:

        multiclass_result = predict_multiclass(
            text=cleaned_text,
            tokenizer=tokenizer,
            multiclass_config=multiclass_config,
            multiclass_labels=multiclass_labels
        )

    except Exception as error:

        st.error(
            "Model multiclass gagal dijalankan."
        )

        st.exception(
            error
        )

        st.stop()

    st.divider()

    st.subheader(
        "Tahap 2 — Jenis Cognitive Distortion"
    )

    st.success(
        (
            "Jenis distorsi yang paling mungkin: "
            f"**{multiclass_result['predicted_label']}**"
        )
    )

    st.metric(
        "Confidence",
        (
            f"{multiclass_result['confidence'] * 100:.2f}%"
        )
    )

    with st.expander(
        "Lihat probabilitas seluruh kelas"
    ):

        st.dataframe(
            multiclass_result[
                "ranked_results"
            ],
            hide_index=True,
            use_container_width=True
        )