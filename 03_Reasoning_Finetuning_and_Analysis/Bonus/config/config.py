import os
import glob

# Workdir determination
WORKDIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

def resolve_paths():
    kaggle_input = "/kaggle/input/datasets/satyagudu"

    # Defaults (Local fallback relative to phase3)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    hindi_data_dir = os.path.join(base_dir, "hindi", "data")
    
    sp_model_path = os.path.join(hindi_data_dir, "hindi_bpe_32000.model")
    train_txt_path = os.path.join(hindi_data_dir, "hindi_train.txt")
    val_txt_path = os.path.join(hindi_data_dir, "hindi_val.txt")
    train_bin_path = os.path.join(WORKDIR, "hindi_train.bin")
    val_bin_path = os.path.join(WORKDIR, "hindi_val.bin")

    if os.path.exists(kaggle_input):
        sp_matches = glob.glob(f"{kaggle_input}/**/hindi_bpe_32000.model", recursive=True)
        if sp_matches:
            sp_model_path = sp_matches[0]

        train_txt_matches = glob.glob(f"{kaggle_input}/**/hindi_train.txt", recursive=True)
        if train_txt_matches:
            train_txt_path = train_txt_matches[0]

        val_txt_matches = glob.glob(f"{kaggle_input}/**/hindi_val.txt", recursive=True)
        if val_txt_matches:
            val_txt_path = val_txt_matches[0]

        train_bin_matches = (
            glob.glob(f"{kaggle_input}/**/train_full.bin", recursive=True)
            or glob.glob(f"{kaggle_input}/**/hindi_train.bin", recursive=True)
        )
        if train_bin_matches:
            train_bin_path = train_bin_matches[0]

        val_bin_matches = (
            glob.glob(f"{kaggle_input}/**/val_full.bin", recursive=True)
            or glob.glob(f"{kaggle_input}/**/hindi_val.bin", recursive=True)
        )
        if val_bin_matches:
            val_bin_path = val_bin_matches[0]

    return sp_model_path, train_txt_path, val_txt_path, train_bin_path, val_bin_path

SP_MODEL_PATH, TRAIN_TXT_PATH, VAL_TXT_PATH, TRAIN_BIN_PATH, VAL_BIN_PATH = resolve_paths()
CKPT_DIR = os.path.join(WORKDIR, "checkpoints", "hindi_no_pos")

MODEL_CONFIG = {
    "vocab_size": 32000,
    "d_model": 448,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 1024,
    "max_seq_len": 512,
    "dropout": 0.1,
    "use_pos_emb": False  # Ablation setting
}

TRAIN_CONFIG = {
    "lr": 3.0e-4,
    "weight_decay": 0.1,
    "betas": (0.9, 0.95),
    "batch_size": 32,
    "grad_accum_steps": 2,
    "max_steps": 10000,
    "warmup_steps": 300,
    "eval_interval": 500,
    "ckpt_interval": 500,
    "grad_clip_norm": 1.0,
}
