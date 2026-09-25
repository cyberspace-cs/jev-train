"""Shared-prefix path encoder for inference (v1: independent paths)."""
import torch


def encode(engine, paths, locations, rows, encoder_impl='auto'):
    """Pack tokenized paths into a batch and run backbone encoding.

    Returns (cand_vecs [Bq,Cmax,H], cand_mask [Bq,Cmax], usage dict).
    """
    torch = engine.torch
    device = engine.device
    tokenizer = engine.tokenizer
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    n_questions = len(rows)
    cmax = max(sum(1 for (q, c) in locations if q == i) for i in range(n_questions))

    length = max(len(p) for p in paths)
    n_paths = len(paths)

    ids = torch.full((n_paths, length), pad, dtype=torch.long, device=device)
    att = torch.zeros_like(ids)
    cand_mask = torch.zeros(n_questions, cmax, dtype=torch.bool, device=device)
    q_index = torch.zeros(n_paths, dtype=torch.long, device=device)
    c_index = torch.zeros(n_paths, dtype=torch.long, device=device)
    ends = torch.zeros(n_paths, dtype=torch.long, device=device)

    path_counts = [0] * n_questions
    for p, (q, c) in enumerate(locations):
        seq = paths[p]
        ids[p, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        att[p, :len(seq)] = 1
        q_index[p] = q
        c_index[p] = c
        ends[p] = len(seq) - 1
        cand_mask[q, c] = True
        path_counts[q] += 1

    batch = {
        "input_ids": ids,
        "attention_mask": att,
        "q_index": q_index,
        "cand_index": c_index,
        "cand_end_pos": ends,
        "cand_mask": cand_mask,
    }

    cand_vecs = engine.model.encode_candidates(batch)

    usage = {
        "encoding_questions": n_questions,
        "encoding_paths": n_paths,
        "encoding_mode": "path",
    }
    return cand_vecs, cand_mask, usage
