# -*- coding: utf-8 -*-
"""
This script is to define discriminative LLMs with classification head
"""

import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, T5Tokenizer, BertJapaneseTokenizer

class pre_train_model_with_DNN_head(nn.Module):
    def __init__(self, config):
        super(pre_train_model_with_DNN_head, self).__init__()
        if(config.ex_dataset[:6] == 'Hazumi'):
            if(config.ex_group == 'Bert'):
                self.pretrained_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
                self.pretrained_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', mecab_kwargs={'mecab_dic': 'unidic'})
            elif(config.ex_group == 'RoBerta'):
                self.pretrained_model = RobertaModel.from_pretrained('rinna/japanese-roberta-base')
                self.pretrained_tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-roberta-base')
        self.input_linear_layer = nn.Linear(config.input_size, config.linear_middle_size[0])
        self.linear_middle_layers = nn.ModuleList([nn.Linear(config.linear_middle_size[i], config.linear_middle_size[i + 1]) for i in range(len(config.linear_middle_size) - 1)])
        self.mapping_layer = nn.Linear(config.linear_middle_size[-1], config.label_size)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.pooling = config.pretrained_pooling
        
    def forward(self, encoded_input):
        model_output_all = self.pretrained_model(**encoded_input, output_hidden_states=True)
        last_hidden_state = model_output_all['last_hidden_state']
        if(self.pooling == 'CLS'):
            linear_input = last_hidden_state[:, 0, :]
        elif(self.pooling == 'MEAN'):
            linear_input = torch.mean(last_hidden_state, dim=1)
        elif(self.pooling == 'SUM'):
            linear_input = torch.sum(last_hidden_state, dim=1)
        output = self.dropout_layer(torch.tanh(self.input_linear_layer(linear_input)))
        for i in range(len(self.linear_middle_layers)):
            output = self.dropout_layer(torch.tanh(self.linear_middle_layers[i](output)))
        output = self.mapping_layer(output)
        return output