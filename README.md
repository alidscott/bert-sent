# Sentiment Analysis on Bert

## What is Bert?
BERT stands for __Bidirectional Encoder Representation for Transformers__ and provides pre-trained representation of language. BERT is state-of-the-art natural language processing model from Google. It can be repurposed for various NLP tasks, such as sentiment analysis which is what I did. In this project, I will also be using DistilBERT which follows a similar architecture as Bert.

"The [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) model was proposed in the blog post Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT, and the paper DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling Bert base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of Bert’s performances as measured on the GLUE language understanding benchmark."

## Installation
Use a package manager to install [Transformer](https://huggingface.co/transformers/index.html) which provides the Bert architectures

### With pip
```bash
pip install transformers
```
### From source
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
```
Install GLUE data which is used to run the BERT models on when fine-tuning
```bash
python download_glue_data.py --data_dir glue_data --tasks all
```

## Running Glue Example
