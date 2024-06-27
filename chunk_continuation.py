import torch

from tqdm import tqdm

def chunk_continuation(model, 
                       tokenizer, 
                       prefix = "My name is", 
                       candidate_set = ["John Doe", "Jane Doe", "John Smith", "Jane Smith"], 
                       suffix = ".",
                       sum=False,
                       verbose=False):
    # encode `prefix` and compute all `past_key_values`.
    input_ids = tokenizer(prefix, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids)
        past_key_values = outputs.past_key_values

    # compute the log-probabilities of all candidates within `candidate_set` given the `prefix`'s `past_key_values`.
    log_probs = []

    for candidate in candidate_set:
        # encode `candidate`
        candidate_ids = tokenizer(candidate, return_tensors="pt").input_ids

        # compute the log probabilities of `candidate` given the prefix's `past_key_values`.
        with torch.no_grad():
            outputs = model(candidate_ids, past_key_values=past_key_values)
            logits = outputs.logits
            # softmax logits to get the log probabilities
            token_log_probs = logits[0, -1, :].log_softmax(dim=-1)
            # select the log-probabilities of the candidate's tokens and sum them.
            if sum:
                log_probs.append(token_log_probs.index_select(0, candidate_ids[0, :]).sum())
            else:
                log_probs.append(token_log_probs.index_select(0, candidate_ids[0, :]).mean())

    if verbose:
        # print the log probabilities of `candidate_set`
        for candidate, log_prob in zip(candidate_set, log_probs):
            print(f"{candidate}: {log_prob.item()}")    

    # compute the categorical distribution from these log probabilities to pick one name.
    probs = torch.softmax(torch.tensor(log_probs), dim=0)
    sampled_cand = torch.multinomial(probs, 1).item()
    if verbose:
        print(f"Sampled candidate: {candidate_set[sampled_cand]}")

    return f'{prefix.strip()} {candidate_set[sampled_cand].strip()}{suffix.strip()}'
