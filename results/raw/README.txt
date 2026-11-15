Live outputs of the current run land here, written by src/rerun_all.py:
per-attempt rows, per-pass summaries, one run manifest per invocation, the
checkpoint-selection log and the smoke-test record.

Nothing in this directory is committed evidence. The outputs the README's
numbers rest on live in results/recorded/<date>/, next to
results/recorded/manifest.json, so that a new run never has to be told apart
from an old one by filename.
