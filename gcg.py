import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets as hf_datasets 
from safetensors.torch import load_file
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate.utils import find_executable_batch_size
from torch import Tensor
from tqdm.autonotebook import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

import torch.nn as nn
from transformers import AutoModel, AutoConfig

class SBERTwithClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        try:
            import flash_attn  # noqa: F401
            self.config._attn_implementation = "flash_attention_2"
        except ImportError:
            pass  # Use default attention implementation
        self.backbone = AutoModel.from_config(self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask
        return self.classifier(sentence_embedding)


# --- Helpers for multi-architecture support ---

def get_word_embeddings(model, model_type):
    """Return the word embedding layer for the given model architecture."""
    if model_type == "sbert":
        return model.backbone.embeddings.word_embeddings
    elif model_type == "bert":
        return model.bert.embeddings.word_embeddings
    elif model_type == "roberta":
        return model.roberta.embeddings.word_embeddings
    elif model_type == "deberta":
        return model.deberta.embeddings.word_embeddings
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_model_dtype(model, model_type):
    """Return the dtype of the model's parameters."""
    return get_word_embeddings(model, model_type).weight.dtype


def forward_with_embeds(model, model_type, inputs_embeds, attention_mask):
    """Compute logits from inputs_embeds for any supported architecture.

    For sbert: backbone → mean pooling → classifier.
    For HF models: full model forward with inputs_embeds (returns logits directly).
    """
    if model_type == "sbert":
        outputs = model.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        lhs = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(lhs.size()).float()
        pooled = torch.sum(lhs * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        return model.classifier(pooled)
    else:
        return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits


# --- Section 1: Dynamic Dataset Loading Fix (From your first file) ---
if not hasattr(hf_datasets, 'load_from_disk'):
    current_dir = os.getcwd()
    # Filter out current dir to avoid circular imports if a local datasets.py exists
    sys.path = [p for p in sys.path if p != current_dir and p != '']
    import importlib
    importlib.reload(hf_datasets)
    from datasets import load_from_disk
else:
    from datasets import load_from_disk

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Enable TF32 for H100/A100 — significant speedup on matmuls and convolutions
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class BERTGCGConfig:
    """Configuration for BERT GCG optimizer."""
    num_steps: int = 250
    search_width: int = 2048
    topk: int = 512
    n_replace: int = 1
    optim_str_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    allow_non_ascii: bool = False
    eval_steps: int = 10
    seed: int | None = None
    verbose: bool = False
    debug_verify: bool = False

def get_nonascii_toks_bert(tokenizer: BertTokenizer, device: str = "cpu"):
    """Get non-ASCII token IDs to filter out."""
    
    def is_ascii(s):
        return s.isascii() and s.isprintable()
    
    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    # Add special tokens
    if tokenizer.cls_token_id is not None:
        ascii_toks.append(tokenizer.cls_token_id)
    if tokenizer.sep_token_id is not None:
        ascii_toks.append(tokenizer.sep_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(ascii_toks, device=device)


def sample_control_bert(
    control_toks: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    temp: float = 1.0,
    not_allowed_tokens: Optional[Tensor] = None,
):
    """Sample candidate token replacements based on gradients.
    
    Args:
        control_toks: Current token IDs [seq_len]
        grad: Gradient w.r.t. token embeddings [seq_len, vocab_size]
        search_width: Number of candidates to generate
        topk: Number of top tokens to consider per position
        temp: Temperature for sampling
        not_allowed_tokens: Token IDs to exclude
    
    Returns:
        Candidate token sequences [search_width, seq_len]
    """
    if not_allowed_tokens is not None:
        # Mask disallowed tokens by setting their gradients to a large value
        grad = grad.clone()
        grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1
    
    # Get top-k token indices with highest (negative) gradients
    # Negative gradient means increasing that token decreases loss
    top_indices = (-grad).topk(topk, dim=1).indices  # [seq_len, topk]
    control_toks = control_toks.to(grad.device)
    
    # Create search_width candidates by replacing tokens at different positions
    original_control_toks = control_toks.repeat(search_width, 1)  # [search_width, seq_len]
    
    # Distribute replacement positions across candidates
    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / search_width, device=grad.device
    ).type(torch.int64)
    
    # Sample from top-k for each position
    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (search_width, 1), device=grad.device),
    )
    
    # Replace tokens at selected positions
    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )
    
    return new_control_toks


class BERTGCGOptimizer:
    """GCG optimizer for BERT classification attacks.
    
    Creates adversarial suffixes that flip sentiment classification using
    gradient-based token optimization.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: BERTGCGConfig,
        device: torch.device = DEVICE,
        model_type: str = "sbert",
    ):
        self.model = model  # torch.compile is counterproductive with dynamic shapes
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.model_type = model_type
        self.embedding_layer = get_word_embeddings(model, model_type)
        self.vocab_size = self.embedding_layer.num_embeddings
        self.hidden_dim = self.embedding_layer.embedding_dim

        # Pre-allocate reusable buffers for torch.compile CUDA graph compatibility
        self._onehot_buffer = None  # Lazily allocated per input shape
        self._full_embeds_buffer = None

        if config.seed is not None:
            torch.manual_seed(config.seed)
    
    def step(
        self,
        input_text: str,
        target_label: int,  # The label we want to flip TO (opposite of original)
    ) -> dict:
        """Optimize adversarial suffix for a single input text.
        
        Args:
            input_text: Original text input
            target_label: Target classification label (0 or 1 for binary)
        
        Returns:
            Dictionary with optimization results
        """
        self._current_input_text = input_text
        print(f"Input text: {input_text}")
        print(f"Target label: {target_label}")
        
        # Tokenize input text (without adversarial suffix)
        input_tokens = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512-20,
            add_special_tokens=True,
        ).to(self.device)
        
        input_ids = input_tokens["input_ids"]  # [1, input_len]
        input_attention_mask = input_tokens["attention_mask"]  # [1, input_len]
        self._current_input_ids = input_ids  # for compute_candidates_loss

        # Get embeddings for input
        input_embeds = self.embedding_layer(input_ids)  # [1, input_len, hidden_dim]
        
        # Initialize adversarial suffix tokens
        optim_str_init = self.config.optim_str_init
        optim_tokens = self.tokenizer(
            optim_str_init,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        )["input_ids"].to(self.device)  # [1, num_optim_tokens]
        
        optim_ids = optim_tokens  # Keep as [1, num_optim_tokens] for consistency
        num_optim_tokens = optim_ids.shape[1]
        
        # Get vocabulary embeddings for gradient computation
        vocab_embeds = self.embedding_layer(
            torch.arange(0, self.vocab_size).long().to(self.device)
        )  # [vocab_size, hidden_dim]
        
        # Filter non-ASCII tokens if needed
        not_allowed_tokens = None
        if not self.config.allow_non_ascii:
            not_allowed_tokens = get_nonascii_toks_bert(self.tokenizer, self.device)
            not_allowed_tokens = torch.unique(not_allowed_tokens)
        
        # Optimization loop
        all_losses = []
        all_suffixes = []
        
        # Pre-allocate stable buffers for the optimization loop
        onehot_shape = (1, num_optim_tokens, self.vocab_size)
        _dtype = get_model_dtype(self.model, self.model_type)
        if self._onehot_buffer is None or self._onehot_buffer.shape != onehot_shape:
            self._onehot_buffer = torch.zeros(
                onehot_shape,
                device=self.device,
                dtype=_dtype,
            )
        total_len = input_embeds.shape[1] + num_optim_tokens
        full_embeds_shape = (1, total_len, self.hidden_dim)
        if self._full_embeds_buffer is None or self._full_embeds_buffer.shape != full_embeds_shape:
            self._full_embeds_buffer = torch.zeros(
                full_embeds_shape,
                device=self.device,
                dtype=_dtype,
            )
        # Pre-compute the static attention mask (doesn't change across steps)
        full_attention_mask = torch.cat(
            [input_attention_mask, torch.ones(1, num_optim_tokens, device=self.device)],
            dim=1
        )

        for i in tqdm(range(self.config.num_steps), desc="GCG Optimization"):
            # ========== Compute coordinate token gradient ========== #
            # Reuse pre-allocated one-hot buffer
            self._onehot_buffer.zero_()
            self._onehot_buffer.scatter_(2, optim_ids.unsqueeze(2), 1.0)
            optim_ids_onehot = self._onehot_buffer.clone().requires_grad_(True)

            # Get embeddings from one-hot (this allows gradient flow)
            optim_embeds = torch.matmul(
                optim_ids_onehot.squeeze(0), vocab_embeds
            ).unsqueeze(0)  # [1, num_optim_tokens, hidden_dim]

            # Reuse pre-allocated full_embeds buffer
            self._full_embeds_buffer[:, :input_embeds.shape[1], :] = input_embeds
            self._full_embeds_buffer[:, input_embeds.shape[1]:, :] = optim_embeds
            full_embeds = self._full_embeds_buffer
            
            # Forward with gradients enabled (eval mode to disable dropout)
            self.model.eval()
            logits = forward_with_embeds(
                self.model, self.model_type, full_embeds, full_attention_mask
            )
            
            # Compute loss: we want to maximize probability of target_label
            # Lower loss = higher probability of target class
            target_labels = torch.tensor(
                [target_label],
                dtype=torch.long,
                device=self.device
            )
            
            loss = F.cross_entropy(logits, target_labels)
            
            # Compute gradients w.r.t. token embeddings
            token_grad = torch.autograd.grad(
                outputs=[loss],
                inputs=[optim_ids_onehot],
                retain_graph=False,
            )[0]  # [1, num_optim_tokens, vocab_size]
            
            # ========== Sample candidates based on gradients ========== #
            sampled_top_indices = sample_control_bert(
                optim_ids.squeeze(0),  # [num_optim_tokens] for sampling
                token_grad.squeeze(0),  # [num_optim_tokens, vocab_size]
                self.config.search_width,
                topk=self.config.topk,
                temp=1.0,
                not_allowed_tokens=not_allowed_tokens,
            )  # [search_width, num_optim_tokens]
            
            # Filter candidates that change after retokenization (vectorized)
            sampled_top_indices_text = self.tokenizer.batch_decode(
                sampled_top_indices, skip_special_tokens=True
            )
            retok = self.tokenizer(
                sampled_top_indices_text,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            retok_ids = retok["input_ids"]  # [search_width, max_retok_len]
            retok_lengths = retok["attention_mask"].sum(dim=1)  # [search_width]

            # A candidate is valid if retokenization produces exactly num_optim_tokens
            # and the token IDs match the original candidate
            length_match = retok_lengths == num_optim_tokens
            if retok_ids.shape[1] >= num_optim_tokens:
                retok_trimmed = retok_ids[:, :num_optim_tokens]
                ids_match = (retok_trimmed == sampled_top_indices).all(dim=1)
            else:
                ids_match = torch.zeros(retok_ids.shape[0], dtype=torch.bool, device=self.device)
            valid_mask = length_match & ids_match
            count = (~valid_mask).sum().item()

            if valid_mask.any():
                sampled_top_indices = sampled_top_indices[valid_mask]
            else:
                if self.config.verbose:
                    print("All removals; defaulting to keeping all")
                count = 0

            if self.config.verbose and count >= self.config.search_width // 2:
                print(f"\nLots of removals: {count}")

            new_search_width = self.config.search_width - count
            
            # ========== Evaluate candidates and select best ========== #
            # Use retokenization to match final evaluation
            loss = find_executable_batch_size(
                self.compute_candidates_loss,
                new_search_width
            )(
                candidate_ids=sampled_top_indices,
                target_label=target_label,
            )
            
            # Update to best candidate (keep argmin on GPU, no .item() sync)
            best_idx = loss.argmin()
            optim_ids = sampled_top_indices[best_idx].unsqueeze(0)  # [1, num_optim_tokens]

            is_eval_step = (i + 1) % self.config.eval_steps == 0 or i == self.config.num_steps - 1
            if is_eval_step:
                # Only materialize loss and decode suffix on eval steps
                current_loss = loss[best_idx].item()
                current_suffix = self.tokenizer.decode(optim_ids[0], skip_special_tokens=True)
                all_losses.append(current_loss)
                all_suffixes.append(current_suffix)

                # Verify by concatenating token IDs directly (matches optimization)
                verify_ids = torch.cat([input_ids, optim_ids], dim=1)
                verify_mask = torch.ones_like(verify_ids)

                with torch.inference_mode():
                    v_logits = forward_with_embeds(
                        self.model, self.model_type,
                        self.embedding_layer(verify_ids),
                        verify_mask.float(),
                    )
                    v_pred = v_logits.argmax(dim=1).item()

                print(f"\nStep {i}: Current Loss: {current_loss:.4f} | Prediction: {v_pred} (Target: {target_label})")

                if v_pred == target_label:
                    print(f"Success found at step {i}! Stopping early.")
                    break

            # Debug verification block (gated behind debug_verify flag)
            if self.config.debug_verify and (self.config.eval_steps == 0 or i == self.config.num_steps - 1):
                if not is_eval_step:
                    current_loss = loss[best_idx].item()
                    current_suffix = self.tokenizer.decode(optim_ids[0], skip_special_tokens=True)
                # Verify using embeddings (same as optimization) to check consistency
                with torch.inference_mode():
                    self.model.eval()

                    optim_embeds_verify = self.embedding_layer(optim_ids)
                    full_embeds_verify = torch.cat([input_embeds, optim_embeds_verify], dim=1)
                    full_attention_mask_verify = torch.cat(
                        [input_attention_mask, torch.ones(1, optim_ids.shape[1], device=self.device)],
                        dim=1
                    )

                    logits_verify = forward_with_embeds(
                        self.model, self.model_type, full_embeds_verify, full_attention_mask_verify
                    )

                    verify_pred_embed = logits_verify.argmax(dim=1).item()
                    verify_probs_embed = F.softmax(logits_verify, dim=1)[0]
                    verify_loss_embed = F.cross_entropy(
                        logits_verify,
                        torch.tensor([target_label], device=self.device, dtype=torch.long)
                    ).item()

                print(f"\n===> Step {i}")
                print(f"===> Suffix: {current_suffix}")
                print(f"===> Optimized Loss (embedding-based): {current_loss:.4f}")
                print(f"===> Verified Loss (embedding-based): {verify_loss_embed:.4f}")
                print(f"===> Verified Prediction (embedding): {verify_pred_embed} (target: {target_label})")
                print(f"===> Verified Probabilities (embedding): {verify_probs_embed.tolist()}")

            # Clean up
            del token_grad
            if 'optim_ids_onehot' in dir():
                del optim_ids_onehot

        best_suffix = all_suffixes[-1] if all_suffixes else self.tokenizer.decode(optim_ids[0], skip_special_tokens=True)
        best_loss = all_losses[-1] if all_losses else float('inf')
        best_ids = optim_ids.squeeze(0)

        print(f"\nBest adversarial suffix: {best_suffix}")
        print(f"Final loss: {best_loss:.4f}")
        
        return {
            "losses": all_losses,
            "suffixes": all_suffixes,
            "best_suffix": best_suffix,
            "best_ids": best_ids,
        }
        
    def compute_candidates_loss(self, search_batch_size: int, candidate_ids: Tensor, target_label: int) -> Tensor:
        all_loss = []
        n_candidates = candidate_ids.shape[0]
        input_ids = self._current_input_ids  # [1, input_len]
        input_len = input_ids.shape[1]
        num_optim_tokens = candidate_ids.shape[1]

        for i in range(0, n_candidates, search_batch_size):
            batch_suffix = candidate_ids[i : i + search_batch_size]  # [bs, num_optim_tokens]
            bs = batch_suffix.shape[0]

            # Concatenate input IDs with suffix IDs directly (no retokenization)
            batch_ids = torch.cat(
                [input_ids.expand(bs, -1), batch_suffix], dim=1
            )  # [bs, input_len + num_optim_tokens]
            batch_mask = torch.ones(bs, input_len + num_optim_tokens, device=self.device)

            with torch.inference_mode():
                batch_embeds = self.embedding_layer(batch_ids)
                logits = forward_with_embeds(
                    self.model, self.model_type, batch_embeds, batch_mask
                )

                target_labels = torch.full((bs,), target_label, device=self.device)
                loss = F.cross_entropy(logits, target_labels, reduction="none")
                all_loss.append(loss)

        return torch.cat(all_loss, dim=0)

    def step_batched(
        self,
        all_examples: list[tuple[str, int]],
        batch_size: int = 8,
    ) -> list[dict]:
        """Optimize adversarial suffixes with dynamic slot refilling.

        When an example succeeds or times out, its batch slot is immediately
        filled with the next pending example, keeping all GPU slots doing
        useful work.

        Args:
            all_examples: Full list of (input_text, target_label) tuples.
            batch_size: Number of concurrent batch slots.

        Returns:
            List of result dicts, one per example (in input order).
        """
        B = min(batch_size, len(all_examples))
        max_input_len = 512 - 20

        # Initialize suffix tokens (shared across all examples)
        init_optim_ids = self.tokenizer(
            self.config.optim_str_init,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        )["input_ids"].to(self.device).squeeze(0)  # [num_optim_tokens]
        num_optim_tokens = init_optim_ids.shape[0]

        # Get vocabulary embeddings (shared)
        vocab_embeds = self.embedding_layer(
            torch.arange(0, self.vocab_size, device=self.device)
        )  # [vocab_size, hidden_dim]

        # Filter non-ASCII tokens
        not_allowed_tokens = None
        if not self.config.allow_non_ascii:
            not_allowed_tokens = get_nonascii_toks_bert(self.tokenizer, self.device)
            not_allowed_tokens = torch.unique(not_allowed_tokens)

        # ========== Pre-allocate batch tensors (fixed shape, slots are mutable) ========== #
        _dtype = get_model_dtype(self.model, self.model_type)
        max_total_len = max_input_len + num_optim_tokens  # 512
        input_embeds = torch.zeros(B, max_input_len, self.hidden_dim, device=self.device, dtype=_dtype)
        slot_input_ids = torch.zeros(B, max_input_len, device=self.device, dtype=torch.long)
        full_attention_mask = torch.zeros(B, max_total_len, device=self.device)
        optim_ids = init_optim_ids.unsqueeze(0).expand(B, -1).clone()  # [B, num_optim_tokens]
        target_labels_t = torch.zeros(B, device=self.device, dtype=torch.long)
        active_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Per-slot tracking
        slot_example_idx = [-1] * B        # which global example index is in each slot
        slot_steps = [0] * B               # how many steps this slot has run
        slot_input_texts = [""] * B        # for eval-step verification
        slot_target_labels = [0] * B       # for eval-step verification
        slot_input_lengths = [0] * B       # actual input token length (no padding)

        # Results storage (indexed by global example index)
        results = [None] * len(all_examples)

        # Queue of pending example indices
        queue = list(range(len(all_examples)))

        def fill_slot(b: int):
            """Fill batch slot b with the next example from the queue."""
            if not queue:
                active_mask[b] = False
                return

            idx = queue.pop(0)
            text, target = all_examples[idx]
            slot_example_idx[b] = idx
            slot_steps[b] = 0
            slot_input_texts[b] = text
            slot_target_labels[b] = target

            # Tokenize WITHOUT padding — suffix will go right after real tokens
            toks = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                max_length=max_input_len,
                truncation=True,
                add_special_tokens=True,
            ).to(self.device)

            input_len = toks["input_ids"].shape[1]
            slot_input_lengths[b] = input_len

            input_embeds[b].zero_()
            input_embeds[b, :input_len] = self.embedding_layer(toks["input_ids"])[0]
            slot_input_ids[b].zero_()
            slot_input_ids[b, :input_len] = toks["input_ids"][0]

            # Mask: 1 for input + suffix positions, 0 for trailing padding
            full_attention_mask[b].zero_()
            full_attention_mask[b, :input_len + num_optim_tokens] = 1.0

            optim_ids[b] = init_optim_ids.clone()
            target_labels_t[b] = target
            active_mask[b] = True

        def finish_slot(b: int, succeeded: bool):
            """Store result for the example in slot b and refill."""
            idx = slot_example_idx[b]
            suffix = self.tokenizer.decode(optim_ids[b], skip_special_tokens=True)
            results[idx] = {
                "best_suffix": suffix,
                "best_ids": optim_ids[b].clone(),
                "succeeded": succeeded,
                "steps": slot_steps[b],
            }
            status = "SUCCESS" if succeeded else "TIMEOUT"
            print(f"  [{status}] Example {idx} after {slot_steps[b]} steps | suffix: {suffix[:60]}...")
            fill_slot(b)

        # ========== Fill initial batch ========== #
        for b in range(B):
            fill_slot(b)

        sw = self.config.search_width
        total_processed = 0
        total_examples = len(all_examples)

        pbar = tqdm(total=total_examples, desc="GCG Attack Queue")

        # ========== Main optimization loop ========== #
        while active_mask.any():
            # --- Batched gradient computation (all active slots) ---
            optim_ids_onehot = torch.zeros(
                (B, num_optim_tokens, self.vocab_size),
                device=self.device,
                dtype=_dtype,
            )
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0)
            optim_ids_onehot.requires_grad_(True)

            optim_embeds = torch.matmul(optim_ids_onehot, vocab_embeds)

            # Build full_embeds with suffix right after input (correct positional embeddings)
            full_embeds_parts = []
            for b in range(B):
                L = slot_input_lengths[b]
                pad_len = max_total_len - L - num_optim_tokens
                part = torch.cat([
                    input_embeds[b, :L],
                    optim_embeds[b],
                    torch.zeros(pad_len, self.hidden_dim, device=self.device, dtype=_dtype),
                ], dim=0)
                full_embeds_parts.append(part)
            full_embeds = torch.stack(full_embeds_parts)

            self.model.eval()
            logits = forward_with_embeds(
                self.model, self.model_type, full_embeds, full_attention_mask
            )

            loss = F.cross_entropy(logits, target_labels_t, reduction="sum")
            token_grad = torch.autograd.grad(
                outputs=[loss], inputs=[optim_ids_onehot], retain_graph=False,
            )[0]

            # --- Per-slot candidate sampling + filtering + embedding-based eval ---
            # Matches reference: sample → filter (decode→re-encode roundtrip) → embed → eval
            all_filtered = []  # list of [valid_count, num_optim_tokens] per slot
            all_cand_loss_list = []  # list of [valid_count] per slot

            eval_bs = min(sw, 1024)

            for b in range(B):
                if not active_mask[b]:
                    all_filtered.append(optim_ids[b].unsqueeze(0))
                    all_cand_loss_list.append(torch.tensor([float('inf')], device=self.device))
                    continue

                sampled = sample_control_bert(
                    optim_ids[b], token_grad[b], sw,
                    topk=self.config.topk, not_allowed_tokens=not_allowed_tokens,
                )  # [sw, num_optim_tokens]

                # Vectorized filtering: decode→re-encode roundtrip check
                sampled_text = self.tokenizer.batch_decode(sampled, skip_special_tokens=True)
                retok = self.tokenizer(
                    sampled_text, add_special_tokens=False, padding=True, return_tensors="pt",
                ).to(self.device)
                retok_ids = retok["input_ids"]
                retok_lengths = retok["attention_mask"].sum(dim=1)
                length_match = retok_lengths == num_optim_tokens
                if retok_ids.shape[1] >= num_optim_tokens:
                    ids_match = (retok_ids[:, :num_optim_tokens] == sampled).all(dim=1)
                else:
                    ids_match = torch.zeros(retok_ids.shape[0], dtype=torch.bool, device=self.device)
                valid_mask = length_match & ids_match

                if valid_mask.any():
                    sampled = sampled[valid_mask]

                all_filtered.append(sampled)

                # Embedding-based candidate evaluation (suffix right after input, no padding gap)
                L = slot_input_lengths[b]
                input_embeds_b = input_embeds[b:b+1, :L]  # [1, L, H] — real tokens only
                n_cands = sampled.shape[0]
                target_b = slot_target_labels[b]

                def _eval_candidates(eval_batch_size, sampled=sampled, input_embeds_b=input_embeds_b,
                                     L=L, target_b=target_b, n_cands=n_cands):
                    losses_b = []
                    with torch.inference_mode():
                        for s in range(0, n_cands, eval_batch_size):
                            e = min(s + eval_batch_size, n_cands)
                            bs = e - s
                            cand_embeds = self.embedding_layer(sampled[s:e])
                            full_embeds_sub = torch.cat(
                                [input_embeds_b.expand(bs, -1, -1), cand_embeds], dim=1
                            )
                            attn_sub = torch.ones(bs, L + num_optim_tokens, device=self.device)
                            lg = forward_with_embeds(
                                self.model, self.model_type, full_embeds_sub, attn_sub
                            )
                            tgt = torch.full((bs,), target_b, device=self.device, dtype=torch.long)
                            losses_b.append(F.cross_entropy(lg, tgt, reduction="none"))
                    return torch.cat(losses_b, dim=0)

                all_cand_loss_list.append(
                    find_executable_batch_size(_eval_candidates, eval_bs)()
                )

            # --- Select best candidate per slot, check success/timeout ---
            for b in range(B):
                if not active_mask[b]:
                    continue

                best_idx = all_cand_loss_list[b].argmin()
                optim_ids[b] = all_filtered[b][best_idx]
                slot_steps[b] += 1

                is_eval = slot_steps[b] % self.config.eval_steps == 0 or slot_steps[b] >= self.config.num_steps

                if is_eval:
                    # Verify by concatenating token IDs directly (matches optimization)
                    L = slot_input_lengths[b]
                    verify_ids = torch.cat([
                        slot_input_ids[b, :L].unsqueeze(0),
                        optim_ids[b].unsqueeze(0),
                    ], dim=1)
                    verify_mask = torch.ones_like(verify_ids, dtype=torch.float)
                    with torch.inference_mode():
                        v_logits = forward_with_embeds(
                            self.model, self.model_type,
                            self.embedding_layer(verify_ids),
                            verify_mask,
                        )
                        v_pred = v_logits.argmax(dim=1).item()

                    if v_pred == slot_target_labels[b]:
                        old_count = total_processed
                        total_processed += 1
                        pbar.update(total_processed - old_count)
                        finish_slot(b, succeeded=True)
                        continue

                # Timeout check
                if slot_steps[b] >= self.config.num_steps:
                    old_count = total_processed
                    total_processed += 1
                    pbar.update(total_processed - old_count)
                    finish_slot(b, succeeded=False)

            del optim_ids_onehot, token_grad

        pbar.close()
        return results


def create_adversarial_suffix_bert_gcg(
    model,
    tokenizer,
    input_text: str,
    original_label: int,
    config: Optional[BERTGCGConfig] = None,
    device: torch.device = DEVICE,
    model_type: str = "sbert",
) -> Tuple[str, dict]:
    """Create adversarial suffix for a classifier using GCG.

    GCG is a white-box method that uses gradients to optimize token replacements.

    Args:
        model: Trained model (SBERTwithClassifier or AutoModelForSequenceClassification)
        tokenizer: Corresponding tokenizer
        input_text: Original text input
        original_label: Original predicted label (0=negative, 1=positive)
        config: GCG configuration (optional)
        device: Device to run on
        model_type: One of 'sbert', 'bert', 'roberta', 'deberta'

    Returns:
        Tuple of (adversarial_suffix, optimization_results_dict)
    """
    if config is None:
        config = BERTGCGConfig()

    # Target label is opposite of original
    target_label = 1 - original_label

    optimizer = BERTGCGOptimizer(model, tokenizer, config, device, model_type=model_type)
    results = optimizer.step(input_text, target_label)
    
    return results["best_suffix"], results

def execute_gcg_attack_sbert(
    model,
    tokenizer,
    test_dataset,
    device=DEVICE,
    max_length=512,
    num_examples=50,
    batch_size=8,
    num_steps=250,
    save_path=None,
    example_indices=None,
    model_type="sbert",
):
    """Run GCG attack evaluation on an SBERT model. Returns metrics dict.

    Uses a fixed evaluation set: the first ``num_examples`` harmful examples
    (label == 1) from the dataset, regardless of whether the model classifies
    them correctly.  Examples the model gets wrong are counted as *failed
    attacks* (i.e. not robust) so that the denominator is always
    ``num_examples`` and results are comparable across models.

    Args:
        model: Trained SBERTwithClassifier (already on device).
        tokenizer: Corresponding tokenizer.
        test_dataset: Dataset with 'text' and 'label' fields (label 1 = harmful).
        device: torch device.
        max_length: Max tokenization length.
        num_examples: Number of harmful examples to evaluate on.
        batch_size: Concurrent GCG batch slots.
        num_steps: GCG optimisation steps per example.
        save_path: If provided, save per-example attack results to this .pt file.
            Includes all examples (succeeded, failed/timeout, and skipped).
        example_indices: If provided, only attack harmful examples at these
            sequential indices (within the harmful-example ordering). Overrides
            num_examples.

    Returns:
        Dict with keys: robust_acc, num_total, num_attacked, num_succeeded,
        num_skipped.
    """
    cfg = BERTGCGConfig(num_steps=num_steps, search_width=2048, topk=512, eval_steps=10)
    optimizer = BERTGCGOptimizer(model, tokenizer, cfg, device, model_type=model_type)

    # Collect harmful examples (fixed set)
    model.eval()
    if example_indices is not None:
        # Collect ALL harmful examples first, then filter by indices
        index_set = set(example_indices)
        max_idx = max(example_indices)
        all_harmful = []  # list of (dataset_idx, text)
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            if item["label"] == 1:
                all_harmful.append((i, item["text"]))
            if len(all_harmful) > max_idx:
                break
        harmful_examples = [all_harmful[j] for j in example_indices if j < len(all_harmful)]
        print(f"GCG eval: using {len(harmful_examples)} pre-selected harmful examples "
              f"(from {len(example_indices)} requested indices)")
    else:
        harmful_examples = []  # list of (dataset_idx, text)
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            if item["label"] == 1:
                harmful_examples.append((i, item["text"]))
            if len(harmful_examples) >= num_examples:
                break

    # Check which ones the model classifies correctly; only attack those
    attack_examples = []  # (text, target_label) for step_batched
    attack_indices = []   # parallel list of (dataset_idx, harmful_idx)
    skipped_indices = []  # harmful indices the model already misclassifies
    num_skipped = 0
    for h_idx, (ds_idx, text) in enumerate(harmful_examples):
        tok = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length).to(device)
        with torch.inference_mode():
            out = model(**tok)
            logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, 'logits') else out)
            pred = logits.argmax(1).item()
        if pred == 1:  # correctly classified as harmful
            attack_examples.append((text, 0))  # target = flip to safe
            attack_indices.append((ds_idx, h_idx))
        else:
            skipped_indices.append((ds_idx, h_idx))
            num_skipped += 1

    num_total = len(harmful_examples)
    num_attacked = len(attack_examples)
    num_succ = 0
    print(f"GCG eval: {num_total} harmful examples, {num_attacked} correctly classified, {num_skipped} skipped")

    if num_attacked > 0:
        results = optimizer.step_batched(attack_examples, batch_size=batch_size)
        num_succ = sum(1 for r in results if r is not None and r["succeeded"])

    # robust_acc denominator is always num_total (fixed set)
    # failed = skipped (misclassified) + succeeded attacks
    robust_acc = 1.0 - (num_succ + num_skipped) / num_total

    print(f"GCG eval: {num_succ}/{num_attacked} attacks succeeded, "
          f"{num_skipped} skipped (misclassified), "
          f"robust_acc={robust_acc:.4f} ({num_total - num_succ - num_skipped}/{num_total})")

    # Save per-example attack results
    if save_path:
        save_data = []
        # Attacked examples (succeeded and failed/timeout)
        for i, (ds_idx, h_idx) in enumerate(attack_indices):
            r = results[i] if (num_attacked > 0 and results is not None) else None
            entry = {
                "dataset_idx": ds_idx,
                "input_text": harmful_examples[h_idx][1],
                "original_label": 1,
                "succeeded": r["succeeded"] if r else False,
                "suffix_text": r["best_suffix"] if r else "",
                "steps": r["steps"] if r else 0,
                "status": "succeeded" if (r and r["succeeded"]) else "failed",
            }
            save_data.append(entry)
        # Skipped examples (model already misclassified)
        for ds_idx, h_idx in skipped_indices:
            save_data.append({
                "dataset_idx": ds_idx,
                "input_text": harmful_examples[h_idx][1],
                "original_label": 1,
                "succeeded": False,
                "suffix_text": "",
                "steps": 0,
                "status": "skipped",
            })
        torch.save(save_data, save_path)
        print(f"Saved {len(save_data)} attack results to {save_path}")

    return {
        "robust_acc": robust_acc,
        "num_total": num_total,
        "num_attacked": num_attacked,
        "num_succeeded": num_succ,
        "num_skipped": num_skipped,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GCG Attack on SBERT/BERT/RoBERTa/DeBERTa")
    parser.add_argument("--base_model_dir", type=str, default="/home/dcheung2/new/ibp_huggingface/sbert/models/sbert_combined_binary", help="Path to the trained model directory")
    parser.add_argument("--model_type", type=str, default="sbert", choices=["sbert", "bert", "roberta", "deberta"], help="Model architecture type")
    parser.add_argument("--num_examples", type=int, default=100, help="Max number of examples to attack")
    parser.add_argument("--dataset_path", type=str, default="/home/dcheung2/new/ibp_huggingface/sbert/datasets/aegis_preprocessed", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of examples to attack simultaneously")
    parser.add_argument("--search_width", type=int, default=2048, help="Number of candidate suffixes per step")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--output", type=str, default="gcg_results.pt", help="Path to save attack results")
    args = parser.parse_args()

    print(f"Loading model from {args.base_model_dir} (type={args.model_type})")

    # --- 1. MODEL SETUP ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)

    if args.model_type == "sbert":
        model = SBERTwithClassifier(args.base_model_dir, num_labels=2).to(DEVICE)

        st_path = os.path.join(args.base_model_dir, "model.safetensors")
        if os.path.exists(st_path):
            print("Loading weights from safetensors...")
            state_dict = load_file(st_path)
        else:
            print("Loading weights from pytorch_model.bin...")
            bin_path = os.path.join(args.base_model_dir, "pytorch_model.bin")
            state_dict = torch.load(bin_path, map_location=DEVICE)

        model.load_state_dict(state_dict, strict=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model_dir).to(DEVICE)

    model.eval()

    # --- 2. DATASET LOADING ---
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)

    dataset = dataset.map(lambda x: {"label": 0 if x["label"] == 0 else 1})

    if "test" in dataset:
        target_dataset = dataset["test"]
    else:
        print("Warning: No test split found, falling back to train.")
        target_dataset = dataset["train"]

    # filter for only harmful to safe attacks
    target_dataset = target_dataset.filter(lambda x: x["label"] == 1)

    # --- 3. Pre-filter examples where model is correct ---
    print(f"\nPre-filtering {min(len(target_dataset), args.num_examples)} examples...")
    attack_examples = []  # (input_text, original_pred) pairs

    for i in range(min(len(target_dataset), args.num_examples)):
        test_example = target_dataset[i]
        input_text = test_example.get("text", test_example.get("sentence"))
        true_label = test_example["label"]

        if not input_text:
            continue

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512 - 20).to(DEVICE)

        with torch.inference_mode():
            outputs = model(**inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            original_pred = logits.argmax(dim=1).item()

        if original_pred != true_label:
            continue

        attack_examples.append((input_text, original_pred))

    total_attacked = len(attack_examples)
    print(f"Found {total_attacked} correctly-classified examples to attack.")

    # --- 4. BATCHED ATTACK WITH SLOT REFILLING ---
    config = BERTGCGConfig(
        num_steps=250,
        search_width=args.search_width,
        topk=min(512, args.search_width),
        verbose=args.verbose,
    )
    optimizer = BERTGCGOptimizer(model, tokenizer, config, DEVICE, model_type=args.model_type)

    # Build full queue of (input_text, target_label) pairs
    all_attack_examples = [(text, 1 - orig_pred) for text, orig_pred in attack_examples]

    try:
        results_list = optimizer.step_batched(all_attack_examples, batch_size=args.batch_size)
    except Exception as e:
        print(f"Attack failed: {e}")
        import traceback; traceback.print_exc()
        results_list = [None] * total_attacked

    # Final verification and summary
    num_succ = 0
    for i, result in enumerate(results_list):
        if result is not None and result["succeeded"]:
            num_succ += 1

    print(f"\nTotal successful attacks: {num_succ} / {total_attacked}")

    # --- 5. SAVE RESULTS ---
    save_data = []
    for i, (input_text, orig_pred) in enumerate(attack_examples):
        result = results_list[i]
        if result is None:
            continue
        save_data.append({
            "example_idx": i,
            "input_text": input_text,
            "original_label": orig_pred,
            "target_label": 1 - orig_pred,
            "suffix_ids": result["best_ids"].cpu(),
            "suffix_text": result["best_suffix"],
            "succeeded": result["succeeded"],
            "steps": result["steps"],
        })

    torch.save(save_data, args.output)
    print(f"Saved {len(save_data)} results to {args.output}")

    # --- 6. VERIFY SAVED FILE ---
    loaded = torch.load(args.output, weights_only=False)
    verify_pass = 0
    verify_fail = 0
    embedding_layer = get_word_embeddings(model, args.model_type)
    for entry in loaded:
        if not entry["succeeded"]:
            continue
        # Verify using token ID concatenation (matches optimization)
        input_toks = tokenizer(
            entry["input_text"], return_tensors="pt", truncation=True,
            max_length=512 - 20, add_special_tokens=True,
        ).to(DEVICE)
        verify_ids = torch.cat([input_toks["input_ids"], entry["suffix_ids"].unsqueeze(0).to(DEVICE)], dim=1)
        verify_mask = torch.ones_like(verify_ids, dtype=torch.float)
        with torch.inference_mode():
            logits = forward_with_embeds(
                model, args.model_type,
                embedding_layer(verify_ids),
                verify_mask,
            )
            pred = logits.argmax(dim=1).item()
        if pred == entry["target_label"]:
            verify_pass += 1
        else:
            verify_fail += 1
            print(f"  [VERIFY FAIL] Example {entry['example_idx']}: pred={pred}, target={entry['target_label']}")

    print(f"Verification: {verify_pass}/{verify_pass + verify_fail} saved attacks still succeed after reload")
