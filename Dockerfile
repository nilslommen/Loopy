FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-sympy python3-pandas python3-tqdm

COPY src/ /src/

CMD ["python3", "/src/evaluate.py", "--tool-cmd", "python3 /src/constant_runtime.py", "--benchmarks", "/benchmarks", "--timeout", "300", "--jobs", "1", "--output-csv", "results_loopy.csv", "--filetype", ".matrix"]
