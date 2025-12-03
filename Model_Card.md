

# Model Card Template (for publishing to Hugging Face Hub)

Model name: <your-model-name>
Model description:
- Base model: e.g., mistralai/mistral-3-small
- Fine-tuning data: high-level summary (domains, sizes), link to dataset provenance
- License: <license>
- Intended use: <intended tasks & limitations>

Training data details:
- Sources and counts
- Preprocessing steps (link to docs/DATASET_PROCESSING.md)
- Filtering and deduplication strategy

Evaluation:
- Metrics used (perplexity, ROUGE, BLEU)
- Evaluation datasets and scripts (link)

Safety & limitations:
- Known failure modes
- Biases & mitigation steps
- Recommended usage and disallowed use cases

How to load:
- Example code to load from HF and run inference
