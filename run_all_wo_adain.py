import subprocess
import itertools

all_domains = [1, 2, 3, 4, 5]
train_combinations = list(itertools.combinations(all_domains, 2))

for train_domains in train_combinations:
    train_set = set(train_domains)
    test_domains = sorted(set(all_domains) - train_set)
    train_domains_str = [str(d) for d in sorted(train_domains)]
    test_domains_str = [str(d) for d in test_domains]

    output_dir = f"results-bpi15-50e{''.join(train_domains_str)}-{''.join(test_domains_str)}"

    cmd = [
        "python", "main_adain.py",
        "--output-dir", output_dir,
        "--models", "graph_transformer",
        "--data-path", "processed_data/bpi15.csv",
        "--train-domains"
    ] + train_domains_str + [
        "--val-domains"
    ] + train_domains_str + [
        "--test-domains"
    ] + test_domains_str + [
        "--num-epochs", "50",
        "--batch-size", "32"
    ]

    print(f"\n👉 Running: {output_dir}")
    print(" ".join(cmd))

    subprocess.run(cmd)
