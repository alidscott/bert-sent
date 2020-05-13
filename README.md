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
### Using DistilBERT
In the jupyter notebook file, _airline-sent.py_, I performed sentiment analysis using the DistilBERT model based off another open [source example](https://github.com/jalammar/jalammar.github.io/tree/master/notebooks/bert). The dataset I used is [Twitter US Airline Sentiment from Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment). It contains tweets on each major airline in the US.

The pre-trained models are imported from the Transformers packages which contains the model I will be using, DistilBERT. First prepare the data which will be passed into the pre-trained DistilBERT model. The dataset is split into a training and testing which will be passed into the logistic regression model.

Evaluating the model, we score an accuracy of around 82%. Our DistilBERT model can be trained to improve this score up to, but not limited to 90%. This process is called fine-tuning. Due to limitations of my own hardware, I am not able to fine-tune this model. More information on fine-tuning and an example is provided below.

## GLUE


### What is GLUE and Fine-Tuning?

"The [General Language Understanding Evaluation](https://mccormickml.com/2019/11/05/GLUE/) benchmark (GLUE) is a collection of datasets used for training, evaluating, and analyzing NLP models relative to one another, with the goal of driving “research in the development of general and robust natural language understanding systems.” The collection consists of nine “difficult and diverse” task datasets designed to test a model’s language understanding, and is crucial to understanding how transfer learning models like BERT are evaluated."

Fine-Tuning allows us to obtain better results on our text classification. We take our pre-trained model, add an untrained layer on top of it, and train it for our classification task. This allows a quicker development while using less data. Fine-tuning is computationally less expensive than training the entire model such as the BERT model.

Based on this [script](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_tf_glue.py), the benchmark results of fine-tuning the library TensorFlow 2.0 Bert model for sequence classification on the MRPC follows below.

<img width="434" alt="becnhmarks" src="https://user-images.githubusercontent.com/14842967/81595312-462ab880-9390-11ea-944d-5eda5bccca4c.png">


## Fine-Tuning Example:

### Installing GLUE data
Install GLUE data by running this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpacking it to some ```$GLUE_DIR```
```bash
python download_glue_data.py --data_dir glue_data --tasks all
```

This fine-tuning example fine-tunes BERT model using the [WikiText2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).
The files that are referred to are: ``` $TRAIN_FILE```, which contains text for training, and ```$TEST_FILE```, which contains text that will be used for evaluation.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_type=bert \
    --model_name_or_path=bert \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE
```


This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run.
