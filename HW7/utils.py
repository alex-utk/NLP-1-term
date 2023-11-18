import torch
import evaluate
import collections
import numpy as np
from tqdm import tqdm


def _preprocess_training(data, inputs):
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = data["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:

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


def _preprocess_validation(data, inputs):
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(data["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def compute_metrics(start_logits, end_logits, features, examples, n_best=20, max_answer_length=30, return_scores=True):
    metric = evaluate.load("squad")
    # список с кортежами вида контекст-вопрос-ответ для визуальной оценки результатов
    whole_results = []

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1:-n_best-1:-1].tolist()
            end_indexes = np.argsort(end_logit)[-1:-n_best-1:-1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (end_index < start_index or
                            end_index - start_index + 1 > max_answer_length):
                        continue

                    answers.append({
                        "text": context[offsets[start_index][0]:offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            answer = best_answer["text"]
        else:
            answer = ''

        predicted_answers.append(
            {"id": str(example_id), "prediction_text": answer}
        )
        whole_results.append((example["context"], example["question"], answer))

    if return_scores:
        theoretical_answers = [{"id": str(ex["id"]), "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)
    else:
        return whole_results


def eval(model, test_loader, test_dataset, raw_test_set, n_best, max_answer_length, return_scores=False):
    start_logits = []
    end_logits = []

    for batch in test_loader:
        with torch.inference_mode():
            outputs = model(**{k: v.cuda() for k, v in batch.items()})

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)

    results = compute_metrics(start_logits, end_logits,
                              test_dataset, raw_test_set,
                              n_best, max_answer_length,
                              return_scores=return_scores)
    return results


def train(model, train_dataloader, eval_dataloader,
          validation_data, dataset_val, optimizer, lr_scheduler,
          n_best, max_answer_length, train_epochs, save_model_path):
    prev_f1_score = 0
    for epoch in tqdm(range(train_epochs)):
        # Training
        model.train()
        for batch in tqdm(train_dataloader):
            outputs = model(**{k: v.cuda() for k, v in batch.items()})
            loss = outputs.loss
            loss.backward() 
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        print("Evaluation!")
        model.eval()
        metrics = eval(model, eval_dataloader, validation_data,
                       dataset_val, n_best, max_answer_length,
                       return_scores=True)
        if metrics['f1'] >= prev_f1_score:
            print(f'{prev_f1_score} -> {metrics["f1"]} SAVING')
            model.save_pretrained(save_model_path)