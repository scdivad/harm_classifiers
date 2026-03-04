import random

from datasets import load_dataset

DATASET_NAME = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
SAVE_PATH = "datasets/aegis_preprocessed"

UTILIZED_LABELS = [
    "Safe",
    "Criminal Planning/Confessions",
    "PII/Privacy",
    "Sexual",
    "Harassment",
    "Guns and Illegal Weapons",
    "Violence",
    "Controlled/Regulated Substances",
]

dataset = load_dataset(
    path=DATASET_NAME,
)

def multi_label_to_single_label(
    candidate_label: str,
    avoid: list = [],
) -> str:
    candidates = candidate_label.split(", ")

    if len(candidates) > 1:
        if candidates[0] not in avoid:
            candidate_label = candidates[0]
        else:
            other_candidates = list(
                filter(
                    lambda x: x not in avoid,
                    candidates,
                )
            )

            if len(other_candidates) > 0:
                candidate_label = random.choice(other_candidates)
            else:
                candidate_label = None
    else:
        candidate_label = candidates[0]
    
    return candidate_label

for split in dataset.keys():
    keep_indices = []
    keep_texts = []
    labels = []

    for i, data in enumerate(dataset[split]):
        keep = True

        data_labels = [
            data[key] for key in [
                "labels_0",
                "labels_1",
                "labels_2",
                "labels_3",
                "labels_4",
            ] if data[key] is not None
        ]

        candidate_label = multi_label_to_single_label(
            candidate_label=data_labels[1],
            avoid=["Safe"],
        )
        other_labels = data_labels[0:1] + data_labels[2:]

        # Replace "Other" with a different label, if possible
        if candidate_label == "Other":
            for label in other_labels:
                if label != "Other":
                    candidate = multi_label_to_single_label(
                        candidate_label=label,
                        avoid=["Safe", "Other"],
                    )

                    if candidate is not None:
                        candidate_label = candidate
                        break
        
        # Make sure "Safe" data is truly safe, otherwise pick an unsafe label
        if candidate_label == "Safe":
            is_safe = True
            
            for label in data_labels:
                if "Safe" not in label:
                    is_safe = False
                    break
            
            if not is_safe:
                not_safe_or_other = list(
                    filter(
                        lambda x: x not in ["Safe", "Other"],
                        data_labels,
                    )
                )

                candidate = None

                while candidate is None and len(not_safe_or_other) > 0:
                    chosen_index = random.randrange(
                        start=0,
                        stop=len(not_safe_or_other),
                    )
                    candidate = multi_label_to_single_label(
                        candidate_label=not_safe_or_other.pop(chosen_index),
                        avoid=["Safe", "Other"],
                    )
                
                candidate_label = candidate
        
        if candidate_label is None or candidate_label in [
            "Other",
            "Needs Caution",
        ]:
            keep = False
        
        if (
            keep and (data["text"] not in keep_texts) and \
                candidate_label in UTILIZED_LABELS
        ):
            keep_indices.append(i)
            keep_texts.append(data["text"])
            labels.append(
                UTILIZED_LABELS.index(candidate_label)
            )
    
    dataset[split] = dataset[split].select(
        indices=keep_indices,
    ).add_column(
        name="label",
        column=labels,
    ).remove_columns(
        column_names=[
            "labels_0",
            "labels_1",
            "labels_2",
            "labels_3",
            "labels_4",
            "num_annotations",
            "id",
            "text_type",
        ],
    )

dataset.save_to_disk(
    dataset_dict_path=SAVE_PATH,
)