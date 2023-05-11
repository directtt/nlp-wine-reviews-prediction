import pandas as pd
import tensorflow as tf
from transformers import DataCollatorWithPadding
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset


def df_to_tf_dataset(df: pd.DataFrame, tokenizer, model) -> tf.data.Dataset:
    """
    Util function to transform pandas dataframe to tensorflow dataset ready for transformer model.

    Args:
        df: input pandas dataframe
        tokenizer: transformer tokenizer
        model: transformer model

    Returns:
        tf.data.Dataset
    """
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    tf_dataset = model.prepare_tf_dataset(
        tokenized_dataset,
        shuffle=False,  # important to stay False for test dataset
        batch_size=16,
        collate_fn=data_collator
    )

    return tf_dataset
