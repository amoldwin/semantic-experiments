from datasets import Dataset
import torch
from transformers import PreTrainedTokenizer
import numpy as np
import torch
import torch.nn.functional as F

exp_name = 'emb_16_featurethenlabel1'



n_features = 10
n_labels = 8
feature_data = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 1, 1],
                          [1, 1, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 1],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 0, 0],
                          [0, 1, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 0, 1]]).T

feature_list = ['can_grow',
                'is_mammal',
                'has_leaves',
                'can_move',
                'has_trunk',
                'can_fly',
                'can_swim',
                'has_stem',
                'is_warmblooded',
                'can_flower']

label_list = ['Goldfish', 'Tuna', 'Robin', 'Canary',
              'Rose', 'Daisy', 'Pine', 'Oak']

 # train_texts = [f"{label} {feature}" 
 #                   for label, row in zip(label_list, feature_data)
 #                   for feature, val in zip(feature_list, row) if val == 1]

# train_texts = [  f"{label} {feature}" if val == 1 else f"{label} NOT {feature}"
#   for label, row in zip(label_list, feature_data)
#   for feature, val in zip(feature_list, row)    ]

train_texts = [f"{feature} {label}" 
                   for label, row in zip(label_list, feature_data)
                   for feature, val in zip(feature_list, row) if val == 1]

eval_texts  = train_texts

# tiny_vocab = {x:i for i,x in enumerate(['[PAD]','[UNK]', 'NOT']+feature_list+label_list)}

tiny_vocab = {x:i for i,x in enumerate(['[PAD]','[UNK]']+feature_list+label_list)}

print(tiny_vocab)


# Tiny vocab
# tiny_vocab = {
#     "[PAD]": 0,
#     "[UNK]": 1,
#     "hello": 2,
#     "world": 3,
#     "goodbye": 4
# }

class TinyVocabTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, pad_token="[PAD]", unk_token="[UNK]", **kwargs):
        self.vocab = vocab
        super().__init__(pad_token=pad_token, unk_token=unk_token, **kwargs)
        self.ids_to_tokens = {v: k for k, v in vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        return text.split()

    def get_vocab(self):
        return self.vocab
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.ids_to_tokens.get(i, self.unk_token) for i in ids]

    def encode(self, text, **kwargs):
        return self.convert_tokens_to_ids(self._tokenize(text))

    def decode(self, token_ids, **kwargs):
        return " ".join(self.convert_ids_to_tokens(token_ids))

tokenizer = TinyVocabTokenizer(tiny_vocab)

# train_texts = ["hello world", "hello hello world", "goodbye world", "hello goodbye"]
# eval_texts  = ["hello world", "goodbye world"]

train_dataset = Dataset.from_dict({"text": train_texts})
eval_dataset  = Dataset.from_dict({"text": eval_texts})

def tokenize_fn(example):
    # example["text"] is a single string
    ids = tokenizer.encode(example["text"])
    return {"input_ids": ids}

# Apply tokenization
train_dataset = train_dataset.map(tokenize_fn, batched=False)
eval_dataset  = eval_dataset.map(tokenize_fn, batched=False)

# Attach labels = input_ids
train_dataset = train_dataset.map(lambda ex: {"labels": ex["input_ids"]}, batched=False)
eval_dataset  = eval_dataset.map(lambda ex: {"labels": ex["input_ids"]}, batched=False)

# Quick sanity check
print("Train sample 0:", train_dataset[0])

def collate_fn(batch):
    max_length = 8
    input_ids = [ex["input_ids"] for ex in batch]
    labels    = [ex["labels"]    for ex in batch]

    padded_input_ids = []
    padded_labels    = []

    for i, l in zip(input_ids, labels):
        i = i[:max_length]
        l = l[:max_length]
        i += [tokenizer.vocab["[PAD]"]] * (max_length - len(i))
        l += [tokenizer.vocab["[PAD]"]] * (max_length - len(l))
        padded_input_ids.append(i)
        padded_labels.append(l)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels":    torch.tensor(padded_labels,    dtype=torch.long),
    }

# Now you can plug `train_dataset`, `eval_dataset`, and `collate_fn` into
# your Trainer(...) and it should avoid the shape error.

import os
import torch
import numpy as np
from transformers import (
    PreTrainedTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import Dataset

vocab_size = tokenizer.vocab_size

config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=64,
    n_embd=32,    # smaller embedding dimension
    n_layer=2,    # fewer layers
    n_head=2      # fewer attention heads
)

# model = GPT2LMHeadModel(config)

tiny_config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=32,   # smaller context window
    n_embd=16,        # embedding dimension
    n_layer=1,        # just 1 transformer block
    n_head=1,         # just 1 attention head
    n_inner=16,       # dimension of the feedforward layer
    # You can turn off or reduce other features if you like:
    n_ctx=32,
    use_cache=False,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1
)
model = GPT2LMHeadModel(tiny_config)

class SaveEmbeddingsCallback(TrainerCallback):
    def __init__(self, tokenizer, output_dir=exp_name+"_embeddings"):
        super().__init__()
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        This method is called at the end of each epoch.
        """
        model = kwargs["model"]
        epoch = state.epoch

        # Get the embeddings
        embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()

        # Save as a NumPy file
        np.save(
            os.path.join(self.output_dir, f"epoch_{int(epoch)}_embeddings.npy"),
            embeddings
        )

        # Optionally, you could also save a readable mapping of token -> embedding
        # We'll show how to save a text file with each token's embedding
        with open(os.path.join(self.output_dir, f"epoch_{int(epoch)}_embeddings.txt"), "w") as f:
            for token, idx in self.tokenizer.vocab.items():
                emb_str = " ".join(f"{v:.4f}" for v in embeddings[idx])
                f.write(f"{token}\t{emb_str}\n")

        print(f"[Callback] Saved embeddings for epoch {epoch}.")


training_args = TrainingArguments(
    output_dir="test_trainer",
    overwrite_output_dir=True,
    num_train_epochs=50000,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",  # Evaluate & call callback each epoch
    logging_strategy="epoch",
    save_strategy="epoch"
)

# Optionally define a compute_metrics for standard LM metrics, e.g., cross-entropy or perplexity.
# For simplicity, let's just report loss here. But you can expand as needed.
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # We can compute average cross-entropy or something more advanced
    # Here, we just return an empty dict or a dummy metric for demonstration
    return {"dummy_metric": 0.0}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    callbacks=[SaveEmbeddingsCallback(tokenizer)],
    compute_metrics=compute_metrics
)


trainer.train()


