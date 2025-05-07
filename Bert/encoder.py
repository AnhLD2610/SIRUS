import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from transformers import RobertaModel
from transformers import BertForMaskedLM
class EncodingModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        if config.model == 'bert':
            self.encoder = BertModel.from_pretrained(config.bert_path).to(config.device)
            self.lm_head = BertForMaskedLM.from_pretrained(config.bert_path).to(config.device).cls
        elif config.model == 'roberta':
            self.encoder = RobertaModel.from_pretrained(config.roberta_path).to(config.device)
            self.encoder.resize_token_embeddings(config.vocab_size)
        if config.tune == 'prompt':
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.bert_word_embedding = self.encoder.get_input_embeddings()
        self.embedding_dim = self.bert_word_embedding.embedding_dim
        self.prompt_lens = config.prompt_len * config.prompt_num
        self.softprompt_encoder = nn.Embedding(self.prompt_lens, self.embedding_dim).to(config.device)
        # initialize prompt embedding
        self._init_prompt()
        self.prompt_ids = torch.LongTensor(list(range(self.prompt_lens))).to(self.config.device)

        self.info_nce_fc = nn.Linear(self.embedding_dim , self.embedding_dim).to(config.device)

    def _init_prompt(self):
        # is is is [e1] is is is [MASK] is is is [e2] is is is
        if self.config.prompt_init == 1:
            prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
            token_embedding = self.bert_word_embedding.weight[2003]
            prompt_embedding[list(range(self.prompt_lens)), :] = token_embedding.clone().detach()
            for param in self.softprompt_encoder.parameters():
                param.data = prompt_embedding # param.data
       
        # ! @ # [e1] he is as [MASK] * & % [e2] just do it  
        elif self.config.prompt_init == 2:
            prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
            ids = [999, 1030, 1001, 2002, 2003, 2004, 1008, 1004, 1003, 2074, 2079,  2009]
            for i in range(self.prompt_lens):
                token_embedding = self.bert_word_embedding.weight[ids[i]]
                prompt_embedding[i, :] = token_embedding.clone().detach()
            for param in self.softprompt_encoder.parameters():
                param.data = prompt_embedding # param.data


    def embedding_input(self, input_ids): # (b, max_len)
        input_embedding = self.bert_word_embedding(input_ids) # (b, max_len, h)
        prompt_embedding = self.softprompt_encoder(self.prompt_ids) # (prompt_len, h)

        for i in range(input_ids.size()[0]):
            p = 0
            for j in range(input_ids.size()[1]):
                if input_ids[i][j] == self.config.prompt_token_ids:
                    input_embedding[i][j] = prompt_embedding[p]
                    p += 1

        return input_embedding


    def forward(self, inputs, is_des=False): # (b, max_length)
        batch_size = inputs['ids'].size()[0]
        tensor_range = torch.arange(batch_size) # (b)     
        pattern = self.config.pattern
        if pattern == 'softprompt' or pattern == 'hybridprompt':
            input_embedding = self.embedding_input(inputs['ids'])
            outputs_words = self.encoder(inputs_embeds=input_embedding, attention_mask=inputs['mask'])[0]
        else:
            outputs_words = self.encoder(inputs['ids'], attention_mask=inputs['mask'])[0] # (b, max_length, h)
            # outputs_words_des = self.encoder(inputs['ids_des'], attention_mask=inputs['mask_des'])[0] # (b, max_length, h)


        # return [CLS] hidden
        if pattern == 'cls' or pattern == 'softprompt':
            clss = torch.zeros(batch_size, dtype=torch.long)
            return outputs_words[tensor_range ,clss] # (b, h)

        # return [MASK] hidden
        elif pattern == 'hardprompt' or pattern == 'hybridprompt':
            masks = []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                try:
                    mask = np.argwhere(ids == self.config.mask_token_ids)[0][0]
                except:
                    mask = 0
                
                masks.append(mask)
            if is_des:
                average_outputs_words = torch.mean(outputs_words, dim=1)
                return average_outputs_words
            else:
                mask_hidden = outputs_words[tensor_range, torch.tensor(masks)] # (b, h)
                return mask_hidden
    
        # return e1:e2 hidden
        elif pattern == 'marker':
            h1, t1 = [], []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                h1_index, t1_index = np.argwhere(ids == self.config.h_ids), np.argwhere(ids == self.config.t_ids)
                h1.append(0) if h1_index.size == 0 else h1.append(h1_index[0][0])
                t1.append(0) if t1_index.size == 0 else t1.append(t1_index[0][0])

            h_state = outputs_words[tensor_range, torch.tensor(h1)] # (b, h)
            t_state = outputs_words[tensor_range, torch.tensor(t1)]

            concerate_h_t = (h_state + t_state) / 2 # (b, h)
            return concerate_h_t
