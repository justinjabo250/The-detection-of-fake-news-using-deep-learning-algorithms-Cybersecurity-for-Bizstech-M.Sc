# The detection of fake news using deep learning algorithms Cybersecurity for Bizstech M.Sc [![Streamlit App](https://img.shields.io/badge/Streamlit-App-yellow)](https://huggingface.co/spaces/Justin-J/The-detection-of-fake-news-using-deep-learning-algorithms-Cybersecurity-for-Bizstech-M.Sc)

![FakeNews](https://github.com/justinjabo250/The-detection-of-fake-news-using-deep-learning-algorithms-Cybersecurity-for-Bizstech-M.Sc/assets/115732734/630d6052-656c-49e6-9573-380dacb38d13)

# 1.0 Introduction

![Fake_News](https://github.com/justinjabo250/The-detection-of-fake-news-using-deep-learning-algorithms-Cybersecurity-for-Bizstech-M.Sc/assets/115732734/848dd682-20cc-4ec4-9fdb-a4befadbf04e)

## What is Fake News Detection?

A natural language processing technology called fake news detection is used to find and categorize erroneous or misleading material in news articles and social media posts. The prevalence of fake news has heightened the demand for automated systems that can scan information for possible deception and flag it. In this project, I looked into how to use Hugging Face to polish a pre-trained fake news detection model and distribute it on the Hugging Face model hub.

# 1.1 Why Hugging Face?

Hugging Face is a platform that offers an extensive collection of tools and resources for machine learning and natural language processing (NLP) operations. Data analysts, developers, and researchers can use its user-friendly interface and a variety of pre-trained models, datasets, and libraries.

Hugging Face provides a wide range of pre-trained models for performing specialized NLP tasks like text categorization, named entity recognition, and sentiment analysis. These models are trained on sizable datasets. These models give you a place to start when doing your analysis and spare you the time and work of creating new models from scratch. I advise you to enroll in [this course](https://huggingface.co/learn/nlp-course/chapter1/1) to learn everything there is to know about natural language processing (NLP) and how to use libraries from the Hugging Face ecosystem for your project.


Please, [go to the website and sign-in](https://huggingface.co/) to access all the features of the platform.
[Read more about Text classification with Hugging Face](https://huggingface.co/tasks/text-classification)

# 1.2 Using GPU Runtime on Google Colab

Before we start with the code, it's important to understand why using [GPU runtime on Google Colab](https://www.youtube.com/watch?v=ovpW1Ikd7pY) is beneficial. GPU stands for Graphical Processing Unit, which is a powerful hardware designed for handling complex graphics and computations. The fake news detection models are Deep Learning-based, so they require significant computational power, such as a GPU, to train efficiently. Please use [Colab](https://colab.research.google.com/) or another GPU cloud provider, or a local machine with an NVIDIA GPU.

In our project, we utilized the GPU runtime on Google Colab to speed up the training process. To access a GPU on Google Colab, all we need to do is select the GPU runtime environment when creating a new notebook. This allows us to take full advantage of the GPU's capabilities and complete our training tasks much faster.

![changing runtime to GPU](https://cdn-images-1.medium.com/max/800/1*1NJACD6Geh69ttzA0F09rQ.gif)

# 2.0 Setup

Now that we have understood the importance of using a GPU, let’s dive into the code. We begin by installing the transformers library, which is a python-based library developed by Hugging Face. This library provides a set of pre-trained models and tools for fine-tuning them. We’ll also install other requirements too.

```shell
!pip install transformers
!pip install datasets
!pip install --upgrade accelerate
!pip install sentencepiece
```

Next, we import the necessary libraries and load the dataset. In this project, we will be using the dataset from [source of fake news dataset]. You can download the dataset here.

```python
import huggingface_hub # Importing the huggingface_hub library for model sharing and versioning
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

# Load the dataset from [source of fake news dataset]
df = pd.read_csv('path/to/dataset.csv')

# Preprocessing steps...
```

# 2.1 Preprocessing

Next, we clean and preprocess the text data. Preprocessing steps may include removing unnecessary characters, tokenization, and normalizing the text. These steps ensure that the data is in a suitable format for training the fake news detection model.

# 2.2 Tokenization

After preprocessing the text data, we need to tokenize it to create numerical representations. Tokenization breaks down the text into smaller units, such as words or subwords, and assigns numerical values to them. This allows the model to process and analyze the text effectively.

```python
checkpoint = "your/pretrained-model"
# define the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenization steps...
```

# 3.0 Training

Now that we have our preprocessed and tokenized data, we can proceed with training the fake news detection model. We'll set the training parameters and initialize the model using the pre-trained checkpoint.

```python
training_args = TrainingArguments(
    "fake_news_detection_trainer",
    num_train_epochs=10,
    load_best_model_at_end=True,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    logging_steps=100,
    per_device_train_batch_size=16,
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Training steps...
```

# 4.0 Next Steps

The model would then need to be evaluated for performance after training, fine-tuned further if necessary, and deployed for actual use. You can investigate several deployment choices, such as creating a web application with Gradio or Streamlit as a framework. By doing so, users would be able to communicate with the model and make predictions about fresh text inputs. I used streamlit for this project.


[![Streamlit App](https://img.shields.io/badge/Streamlit-App-yellow)](https://huggingface.co/spaces/Justin-J/The-detection-of-fake-news-using-deep-learning-algorithms-Cybersecurity-for-Bizstech-M.Sc)

![fakenewsappgif](https://github.com/Gitjohhny/FakeNews-Detection-with-deep-learning-models/assets/110716071/18f793bb-d507-4476-b4c8-374c7a7a2809)

# 5.0 Conclusion

In conclusion, utilizing Hugging Face, we have improved a pre-trained model for spotting false news. You can duplicate the procedure and modify it for your own false news detection initiatives by adhering to the methods described in this README.


# 5.1 Resources

1. [Quick intro to NLP](https://www.youtube.com/watch?v=CMrHM8a3hqw)
2. [Getting Started With Hugging Face in 15 Minutes](https://www.youtube.com/watch?v=QEaBAZQCtwE)
3. [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs)
4. [Fine-Tuning-DistilBert - Hugging Face Transformer for Poem Sentiment Prediction | NLP](https://www.youtube.com/watch?v=zcW2HouIIQg)
5. [Introduction to NLP: Playlist](https://www.youtube.com/playlist?list=PLM8wYQRetTxCCURc1zaoxo9pTsoov3ipY)
