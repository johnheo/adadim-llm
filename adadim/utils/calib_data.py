import torch
from datasets import load_dataset

def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    print('>> loading...', data)
    if data == "pileval":
        # dataset = load_dataset("json", data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst", split="train")
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif data == 'gsm8k':
        dataset = load_dataset("gsm8k", 'main', split='train')
    elif data == 'mbpp':
        dataset = load_dataset('mbpp', 'full', split='train') # full, sanitized
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0

    key_map = {
        'pileval': 'text',
        'gsm8k': 'answer', #[question, answer]
        'mbpp': 'code', #[prompt, code]
    }
    key = key_map[data]
    for data in dataset:
        line = data[key]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    print(' * num samples collected: ', n_run)
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks with block_size {block_size}")
    
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]
