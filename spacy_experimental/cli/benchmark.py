from typing import Iterable, List, Optional
import itertools
import random
from itertools import islice
import numpy
from pathlib import Path
from spacy import Language, util
from spacy.cli import app
from spacy.tokens import Doc
from spacy.training import Corpus
from thinc.api import require_gpu
from thinc.util import gpu_is_available
import time
from typer import Argument as Arg, Option
from wasabi import Printer


def infinite_shuf(docs: List[Doc]):
    while True:
        yield random.choice(docs)


@app.command("benchmark")
def benchmark_cli(
    model: str = Arg(..., help="Model name or path"),
    data_path: Path = Arg(
        ..., help="Location of binary evaluation data in .spacy format", exists=True
    ),
    batch_size: Optional[int] = Option(
        None, "--batch-size", "-b", min=1, help="Override the pipeline batch size"
    ),
    use_gpu: int = Option(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),
    bench_epochs: int = Option(
        3,
        "--iter",
        "-i",
        min=1,
        help="Number of iterations over the data for benchmarking",
    ),
    warmup_epochs: int = Option(
        3, "--warmup", "-w", min=0, help="Number of iterations over the data for warmup"
    ),
):
    """
    Benchmark a pipeline. Expects a loadable spaCy pipeline and benchmark
    data in the binary .spacy format.
    """
    setup_gpu(use_gpu=use_gpu, silent=False)

    nlp = util.load_model(model)
    corpus = Corpus(data_path)
    docs = [eg.predicted for eg in corpus(nlp)]
    n_tokens = count_tokens(docs)

    times, wps = warmup(nlp, docs, warmup_epochs, batch_size)
    print(f"Mean batch speed: {numpy.mean(wps)}")
    print(f"Mean batch time: {numpy.mean(times)}")

    best = benchmark(nlp, docs, bench_epochs, batch_size)

    print(
        "Best of %d runs: %.3fs %.0f, words/s" % (bench_epochs, best, n_tokens / best)
    )


def annotate(nlp: Language, docs: Iterable[Doc], batch_size: Optional[int]) -> None:
    docs = nlp.pipe(docs, batch_size=batch_size)
    wps = []
    times = []
    while True:
        try:
            doc_start = time.time()
            batch_docs = list(
                islice(docs, batch_size if batch_size else nlp.batch_size)
            )
            if len(batch_docs) == 0:
                break
            elapsed = time.time() - doc_start
            n_tokens = count_tokens(batch_docs)
            times.append(elapsed)
            wps.append(n_tokens / elapsed)

        except StopIteration:
            break

    return times, wps


def benchmark(
    nlp: Language, docs: Iterable[Doc], bench_epochs: int, batch_size: Optional[int]
) -> None:
    best = float("inf")
    for _ in range(bench_epochs):
        start = time.time()
        annotate(nlp, docs, batch_size)
        end = time.time()
        run = end - start
        if run < best:
            best = run
    return best


def count_tokens(docs: Iterable[Doc]) -> int:
    count = 0
    for doc in docs:
        for _ in doc:
            count += 1
    return count


def warmup(
    nlp: Language, docs: List[Doc], warmup_epochs: int, batch_size: Optional[int]
) -> None:
    docs = itertools.chain(*[iter(docs) for _ in range(warmup_epochs)])
    return annotate(nlp, docs, batch_size)


# Verbatim copy from spacy.cli._util. Remove after possible
# spaCy integration.
def setup_gpu(use_gpu: int, silent=None) -> None:
    """Configure the GPU and log info."""
    if silent is None:
        local_msg = Printer()
    else:
        local_msg = Printer(no_print=silent, pretty=not silent)
    if use_gpu >= 0:
        local_msg.info(f"Using GPU: {use_gpu}")
        require_gpu(use_gpu)
    else:
        local_msg.info("Using CPU")
        if gpu_is_available():
            local_msg.info("To switch to GPU 0, use the option: --gpu-id 0")
