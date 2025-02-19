from utils import (
    handling_dataset,
    train_model,
    compute_metrics,
)
from faissDatabase import vector_database, get_embeddings
import torch
from transformers import pipeline


if __name__ == "__main__":
    # # Define parameters
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    #
    # train_datasets, val_datasets, raw_datasets = handling_dataset()
    #
    # # Training
    # trainer = train_model(train_datasets, val_datasets)
    #
    # # Evaluation
    # predictions, _, _ = trainer.predict(val_datasets)
    #
    # start_logits, end_logits = predictions
    #
    # results = compute_metrics(
    #     start_logits, end_logits, val_datasets, raw_datasets["validation"]
    # )

    PIPELINE_NAME = "question-answering"
    MODEL_NAME = "./output/"
    pipe = pipeline(PIPELINE_NAME, model=MODEL_NAME)

    # # Testing
    # INPUT_QUESTION = "What is my name?"
    # INPUT_CONTEXT = "My name is AI VIETNAME and I live in Vietnam"
    # pipe(questions=INPUT_QUESTION, context=INPUT_CONTEXT)

    embeddings_dataset, EMBEDDING_COLUMN = vector_database()

    input_questions = "When did Beyonce start become popular?"

    input_question_embeddings = get_embeddings([input_questions])
    input_question_embeddings = input_question_embeddings.cpu().detach().numpy()

    TOP_K = 5
    scores, samples = embeddings_dataset.get_nearest_examples(
        EMBEDDING_COLUMN, input_question_embeddings, k=TOP_K
    )

    for idx, score in enumerate(scores):
        question = samples["question"][idx]
        context = samples["context"][idx]
        answer = pipe(questions=question, context=context)
        print(f"Top {idx + 1}\tScore: {score}")
        print(f"context: {context}")
        print(f"answer: {answer}")
        print()
