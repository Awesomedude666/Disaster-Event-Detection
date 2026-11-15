import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from src.config import XLM_MODEL, BATCH_SIZE, MAX_SEQ_LEN


def get_xlm_roberta_embeddings(
    texts: list,
    model_name: str = XLM_MODEL,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
) -> np.ndarray:
    """
    Step 1: Generate multilingual sentence embeddings via attention-mask weighted mean pooling.
    """
    print(f"\n[STEP 1] Generating XLM-RoBERTa embeddings ...")
    print(f"  Model  : {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    for start in tqdm(range(0, len(texts), batch_size), desc="  Encoding batches"):
        batch = texts[start : start + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()

        all_embeddings.append(mean_pooled)

    embeddings = np.vstack(all_embeddings)
    print(f"  Embedding shape : {embeddings.shape}")
    return embeddings