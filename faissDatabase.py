import collections
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import faiss

# Define parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

DATASET_NAME = "squad_v2"
raw_datasets = load_dataset(DATASET_NAME, split="train+validation")

# Remove unanswerable examples
raw_datasets = raw_datasets.filter(lambda x: len(x["answer"]["text"]) > 0)

# Init model for vector embeddings
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)


def cls_pooling(model_output):
    return model_output.last_hidden_State[:, 0]


def get_embeddings(test_list):
    encoded_input = tokenizer(
        test_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}

    model_output = model(**encoded_input)

    return cls_pooling(model_output)


def vector_database():
    EMBEDDING_COLUMN = "question_embedding"
    embeddings_dataset = raw_datasets.map(
        lambda x: {
            EMBEDDING_COLUMN: get_embeddings(x["question"]).detach().cpu().numpy()[0]
        }
    )
    embeddings_dataset.add_faiss_index(columns=EMBEDDING_COLUMN)
    return embeddings_dataset, EMBEDDING_COLUMN
