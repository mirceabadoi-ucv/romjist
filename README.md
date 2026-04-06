
# Layered Explanations for Romanian Credit-Eligibility Chatbots

This repository contains pipeline for the paper "Layered Explanations for Romanian Credit-Eligibility Chatbots".

## Quick Start
1. Install dependencies: pip install -r requirements.txt
2. Ensure german.data is in the root folder.
3. Run the pipeline: python layered_pipeline.py

## Methodology Replication
The code replicates the following paper sections:
- *Sec 3.2/3.3*: Verbalization of Statlog (German Credit) and Synthetic corpus generation.
- *Sec 4.3*: Three-layer XAI (Token Attribution, Natural Language Rationales, and Symbolic Rule Traces via Decision Trees).
- *Sec 5.4*: Faithfulness (top-k deletion) and Robustness (diacritic removal and 5% character noise).
