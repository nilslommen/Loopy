# install tqdm and pandas

import argparse
import os
import subprocess
import concurrent.futures
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import csv
import time

def run_tool_on_benchmark(tool_cmd, benchmark_path, timeout, store_output, output_dir):
    benchmark_name = os.path.basename(benchmark_path)
    full_cmd = f"{tool_cmd} {benchmark_path}"

    start_time = time.monotonic()
    try:
        result = subprocess.run(full_cmd, shell=True, timeout=timeout, capture_output=True, text=True)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        elapsed = time.monotonic() - start_time

        if result.returncode != 0:
            status = f"Error (exit {result.returncode})"
            output_line = stderr.splitlines()[0] if stderr else "Crash"
        else:
            output_line = stdout.splitlines()[0] if stdout else "No Output"
            status = "OK"

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start_time
        status = "TIMEOUT"
        output_line = "TIMEOUT"
        stdout = ""
    except Exception as e:
        elapsed = time.monotonic() - start_time
        status = f"EXCEPTION: {str(e)}"
        output_line = "EXCEPTION"
        stdout = ""

    # Optionally store the full output
    if store_output:
        out_file = Path(output_dir) / f"{benchmark_name}.log"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(stdout)

    return benchmark_name, output_line, status, elapsed

def main():
    parser = argparse.ArgumentParser(description="Evaluate tool on benchmarks using Docker")
    parser.add_argument("--tool-cmd", required=True, help="Command to run the tool in Docker")
    parser.add_argument("--benchmarks", required=True, help="Path to benchmark folder")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per benchmark (in seconds)")
    parser.add_argument("--store-output", action='store_true', help="Store full output in log files")
    parser.add_argument("--jobs", type=int, default=6, help="Number of parallel jobs")
    parser.add_argument("--output-csv", default="results.csv", help="Path to output CSV file")
    parser.add_argument("--output-dir", default="outputs", help="Directory to store full outputs")
    parser.add_argument("--filetype", default=".txt", help="File extension for benchmark files (e.g., .txt, .json)")

    args = parser.parse_args()

    # Ensure output directory exists if storing full output
    if args.store_output:
        os.makedirs(args.output_dir, exist_ok=True)

    benchmark_files = benchmark_files = sorted([
                        str(p) for p in Path(args.benchmarks).rglob(f"*{args.filetype}")
                            if p.is_file()
                    ])

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_tool_on_benchmark, args.tool_cmd, path, args.timeout, args.store_output, args.output_dir): path
            for path in benchmark_files
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = future.result()
                rel_name = os.path.relpath(futures[future], args.benchmarks).replace(os.sep, '/')

                _, output_line, status, elapsed = result
                results.append((rel_name, output_line, status, elapsed))
            except Exception as e:
                name = os.path.relpath(futures[future], args.benchmarks).replace(os.sep, '/')
                results.append((name, "Crash", f"Exception: {str(e)}", 0.0))

    # Write results to CSV
    df = pd.DataFrame(results, columns=["Benchmark", "Result", "Status", "Time"])
    df = df.sort_values(by="Benchmark")
    csv_path = os.path.join(args.output_dir, args.output_csv)
    csv_path = Path(args.output_dir) / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"Evaluation complete. Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
