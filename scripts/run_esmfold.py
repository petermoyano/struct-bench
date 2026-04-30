"""ESMFold benchmark runner.

Runs ESMFold (via HuggingFace transformers) on a FASTA file at a specified
numerical precision and writes per-variant results to CSV.

Usage:
    python scripts/run_esmfold.py --precision fp32
    python scripts/run_esmfold.py --precision bf16
    python scripts/run_esmfold.py --config path/to/config.yaml --precision fp32
"""

import argparse
import csv
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_fasta(fasta_path: str) -> list[tuple[str, str]]:
    """Parse a FASTA file into a list of (header, sequence) tuples."""
    sequences = []
    header = None
    seq_lines: list[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    sequences.append((header, ''.join(seq_lines)))
                header = line[1:]
                seq_lines = []
            elif line:
                seq_lines.append(line)

    if header is not None:
        sequences.append((header, ''.join(seq_lines)))

    return sequences


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    print('[WARN] No CUDA GPU detected — running on CPU (very slow)')
    return torch.device('cpu')


def get_hardware_info(device: torch.device) -> dict:
    info = {
        'device': str(device),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_vram_mb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e6)
        info['cuda_version'] = torch.version.cuda
    return info


def load_model(device: torch.device, config: dict):
    """Load ESMFold via HuggingFace transformers."""
    from transformers import AutoTokenizer, EsmForProteinFolding

    print('[INFO] Loading ESMFold model (facebook/esmfold_v1)...')
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained('facebook/esmfold_v1')
    model = EsmForProteinFolding.from_pretrained(
        'facebook/esmfold_v1',
        low_cpu_mem_usage=True,
    )

    esm_cfg = config.get('esmfold', {})

    if esm_cfg.get('half_stem', True) and device.type == 'cuda':
        model.esm = model.esm.half()

    model = model.to(device)
    model.eval()

    chunk_size = esm_cfg.get('chunk_size', 64)
    if chunk_size and hasattr(model, 'trunk') and hasattr(model.trunk, 'set_chunk_size'):
        model.trunk.set_chunk_size(chunk_size)

    load_time = time.time() - t0
    print(f'[INFO] Model loaded in {load_time:.1f}s')

    return model, tokenizer


def predict_single(
    model,
    tokenizer,
    sequence: str,
    device: torch.device,
    precision: str,
    max_recycles: int,
) -> dict:
    """Run inference on a single sequence. Returns metrics dict."""
    tokenized = tokenizer(
        [sequence],
        return_tensors='pt',
        add_special_tokens=False,
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.time()

    if precision == 'bf16' and device.type == 'cuda':
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(**tokenized, num_recycles=max_recycles)
    else:
        with torch.no_grad():
            output = model(**tokenized, num_recycles=max_recycles)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    inference_time = time.time() - t0

    plddt = output.plddt[0, 1:-1]
    mean_plddt = plddt.mean().item()

    peak_memory_mb = 0.0
    if device.type == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    return {
        'mean_plddt': mean_plddt,
        'inference_time_s': round(inference_time, 3),
        'peak_memory_mb': round(peak_memory_mb, 1),
    }


def run_benchmark(config: dict, precision: str):
    fasta_path = config['fasta_path']
    output_dir = Path(config.get('output_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    esm_cfg = config.get('esmfold', {})
    max_recycles = esm_cfg.get('max_recycles', 4)

    sequences = parse_fasta(fasta_path)
    print(f'[INFO] Loaded {len(sequences)} sequences from {fasta_path}')

    device = detect_device()
    hw_info = get_hardware_info(device)
    print(f'[INFO] Device: {hw_info.get("gpu_name", "CPU")}')

    model, tokenizer = load_model(device, config)

    csv_path = output_dir / f'esmfold_{precision}.csv'
    meta_path = output_dir / f'esmfold_{precision}_meta.yaml'

    fieldnames = [
        'variant_id', 'model', 'precision', 'mean_plddt',
        'inference_time_s', 'peak_memory_mb', 'sequence_length', 'status',
    ]

    results = []

    for idx, (header, seq) in enumerate(sequences):
        label = f'[{idx + 1}/{len(sequences)}] {header}'
        try:
            metrics = predict_single(model, tokenizer, seq, device, precision, max_recycles)
            row = {
                'variant_id': header,
                'model': 'esmfold',
                'precision': precision,
                'mean_plddt': round(metrics['mean_plddt'], 2),
                'inference_time_s': metrics['inference_time_s'],
                'peak_memory_mb': metrics['peak_memory_mb'],
                'sequence_length': len(seq),
                'status': 'ok',
            }
            print(f'  {label}  pLDDT={row["mean_plddt"]:.1f}  t={row["inference_time_s"]:.1f}s  mem={row["peak_memory_mb"]:.0f}MB')
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {
                'variant_id': header,
                'model': 'esmfold',
                'precision': precision,
                'mean_plddt': float('nan'),
                'inference_time_s': float('nan'),
                'peak_memory_mb': float('nan'),
                'sequence_length': len(seq),
                'status': 'oom',
            }
            print(f'  {label}  OOM — skipped')
        except Exception as e:
            row = {
                'variant_id': header,
                'model': 'esmfold',
                'precision': precision,
                'mean_plddt': float('nan'),
                'inference_time_s': float('nan'),
                'peak_memory_mb': float('nan'),
                'sequence_length': len(seq),
                'status': f'error: {e}',
            }
            print(f'  {label}  ERROR: {e}')

        results.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    ok_count = sum(1 for r in results if r['status'] == 'ok')
    ok_results = [r for r in results if r['status'] == 'ok']

    meta = {
        'model': 'esmfold',
        'precision': precision,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'fasta_path': str(fasta_path),
        'total_sequences': len(sequences),
        'successful': ok_count,
        'failed': len(sequences) - ok_count,
        'config': config.get('esmfold', {}),
        'hardware': hw_info,
    }
    if ok_results:
        times = [r['inference_time_s'] for r in ok_results]
        plddts = [r['mean_plddt'] for r in ok_results if not math.isnan(r['mean_plddt'])]
        meta['summary'] = {
            'mean_plddt_avg': round(sum(plddts) / len(plddts), 2) if plddts else None,
            'inference_time_avg_s': round(sum(times) / len(times), 3),
            'inference_time_total_s': round(sum(times), 1),
        }

    with open(meta_path, 'w') as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print(f'\n[DONE] Results: {csv_path}')
    print(f'[DONE] Metadata: {meta_path}')
    if ok_results:
        s = meta['summary']
        print(f'[DONE] {ok_count}/{len(sequences)} succeeded | avg pLDDT={s["mean_plddt_avg"]} | avg time={s["inference_time_avg_s"]}s | total={s["inference_time_total_s"]}s')


def main():
    parser = argparse.ArgumentParser(description='ESMFold benchmark runner')
    parser.add_argument('--precision', type=str, required=True, choices=['fp32', 'bf16'],
                        help='Numerical precision for inference')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file')
    args = parser.parse_args()

    config = load_config(args.config)
    print(f'=== ESMFold Benchmark | precision={args.precision} ===')
    run_benchmark(config, args.precision)


if __name__ == '__main__':
    main()
