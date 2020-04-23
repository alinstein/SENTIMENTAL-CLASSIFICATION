from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW, XLNetConfig
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
import torch


def model_sel(model_type='Bert'):
      if model_type=='XLNet':
          print("Model is XLNet")
          tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
          model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased',
                              num_labels = 5, # The number of output labels--2 for binary classification.
                              # You can increase this for multi-class tasks.   
                              output_attentions = False, # Whether the model returns attentions weights.
                              output_hidden_states = False, # Whether the model returns all hidden-states.
                            )
          model.cuda()


      elif model_type=='Bert':
          print("Model is Bert")
          # Load BertForSequenceClassification, the pretrained BERT model with a single 
          # linear classification layer on top. 
          tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
          model = BertForSequenceClassification.from_pretrained(
              "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
              num_labels = 5, # The number of output labels--2 for binary classification.
                              # You can increase this for multi-class tasks.   
              output_attentions = False, # Whether the model returns attentions weights.
              output_hidden_states = False, # Whether the model returns all hidden-states.
          )

          # Tell pytorch to run this model on the GPU.
          model.cuda()


      return model, tokenizer