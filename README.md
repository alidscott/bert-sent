# Sentiment Analysis on Bert

## What is Bert?
BERT stands for __Bidirectional Encoder Representation for Transformers__ and provides pre-trained representation of language. BERT is state-of-the-art natural language processing model from Google. It can be repurposed for various NLP tasks, such as sentiment analysis which is what I did. In this project, I will also be using DistilBERT which follows a similar architecture as Bert.

"The [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) model was proposed in the blog post Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT, and the paper DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling Bert base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of Bert’s performances as measured on the GLUE language understanding benchmark."

## Installation

Use a package manager to install [Transformer](https://huggingface.co/transformers/index.html) which provides the Bert architectures. It's an open source library built for NLP researchers seeking to use/study/extend large-scale transformers models.

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
### GLUE
### What is GLUE?

The [General Language Understanding Evaluation](https://mccormickml.com/2019/11/05/GLUE/) benchmark (GLUE) is a collection of datasets used for training, evaluating, and analyzing NLP models relative to one another, with the goal of driving “research in the development of general and robust natural language understanding systems.” The collection consists of nine “difficult and diverse” task datasets designed to test a model’s language understanding, and is crucial to understanding how transfer learning models like BERT are evaluated.

Install GLUE data by running this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpacking it to some __$GLUE_DIR__
```bash
python download_glue_data.py --data_dir glue_data --tasks all
```

## Running a Glue Example

Quick Benchmarks on running the GLUE script

<img width="434" alt="becnhmarks" src="https://user-images.githubusercontent.com/14842967/81595312-462ab880-9390-11ea-944d-5eda5bccca4c.png">
