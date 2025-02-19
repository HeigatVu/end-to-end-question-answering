from typing import List, Dict
from tqdm.auto import tqdm

from datasets import load_dataset
import collections

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)
import numpy as np
import evaluate

# Parameters

MAX_LENGTH = 384
STRIDE = 128
MODEL_NAME = "distilbert-base-uncased"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Download
DATASET_NAME = "squad_v2"
raw_datasets = load_dataset(DATASET_NAME)

N_BEST = 20
MAX_ANS_LENGTH = 30


# preprocessing data
def preprocessing_training_data(examples: Dict) -> Dict[str, List[int]]:

    # Extracting Question from examples
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Extracting mapping out of inputs
    offset_mapping = inputs.pop("offset_mapping")

    sample_map = inputs.pop("overflow_to_sample_mapping")

    # Extracting answers from examples
    answers = examples["answers"]

    # Init list start and end of answers
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        # Taking index of sample related to offset
        sample_idx = sample_map[i]
        sequence_ids = inputs.sequence_ids(i)

        # Finding start and end of context in the sameple sequence
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Extacting answers for sameple
        answer = answers[sample_idx]

        # Handling empty answers
        if len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Saving start and end position in context
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])

            # Handling answers' posititon out of context range
            if (
                offset[context_start][0] > start_char
                or offset[context_end][1] < end_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Choosing start and end position in context based on offset_mapping
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

    return inputs


def preprocessing_valiadation_examples(examples: Dict) -> Dict[str, List[int]]:
    # Extracting Question from examples
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Extracting mapping out of inputs
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    # Identifying ids of examples and adjust mapping offset
    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]

        # Remove inappropriate offset not matching with sequence_ids
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    # Extracting answers from examples
    inputs["example_id"] = example_ids

    return inputs


def handling_dataset():
    # load datasets
    train_datasets = raw_datasets["train"].map(
        preprocessing_training_data,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    val_datasets = raw_datasets["validation"].map(
        preprocessing_valiadation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    return train_datasets, val_datasets, raw_datasets


def train_model(train_dataset, validation_dataset):
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    args = TrainingArguments(
        output_dir="./output",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    return trainer


metric = evaluate.load("squad_v2")


def compute_metrics(start_logits, end_logits, features, examples):
    # Create a default dictionary
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -N_BEST - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -N_BEST - 1 : -1].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue

                        if end_index - start_index + 1 > MAX_ANS_LENGTH:
                            continue

                        text = context[offsets[start_index][0] : offsets[end_index][1]]
                        logit_score = start_logits[start_index] + end_logits[end_index]
                        answer = {
                            "text": text,
                            "logit_score": logit_score,
                        }
                        answers.append(answer)

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])

            answer_dict = {
                "id": example_id,
                "prediction_text": best_answer["text"],
                "no_answer_probability": 1 - best_answer["logit_score"],
            }
        else:

            answer_dict = {
                "id": example_id,
                "prediction_text": "",
                "no_answer_probability": 1.0,
            }
        predicted_answers.append(answer_dict)

    theoretical_answers = [{"id": ex["id"], "answer": ex["answers"]} for ex in examples]

    return metric.compute(
        predictions=predicted_answers,
        references=theoretical_answers,
    )


if __name__ == "__main__":

    # Testing with preprocessing datasets
    train_datasets = raw_datasets["train"].map(
        preprocessing_training_data,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    print(train_datasets)
    print(train_datasets["start_positions"][:2])
    print(train_datasets["end_positions"][:2])
    print(train_datasets["input_ids"][:2])
    print(train_datasets["attention_mask"][:2])

    # Testing with preprocessing validation datasets
    val_datasets = raw_datasets["validation"].map(
        preprocessing_valiadation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    print(val_datasets)
    print(val_datasets["input_ids"][:2])
    print(val_datasets["attention_mask"][:2])
    print(val_datasets["example_id"][:2])

    print(
        f"raw_datasets: {len(raw_datasets['validation'])}, validation_dataset: {len(val_datasets)}"
    )
