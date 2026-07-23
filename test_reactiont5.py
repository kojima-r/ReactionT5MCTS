import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig, PreTrainedModel
from transformers import PreTrainedModel, T5Config, T5ForConditionalGeneration

from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig, PreTrainedModel
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)


# ReactionT5v2 retrosynthesis
# https://huggingface.co/sagawa/ReactionT5v2-retrosynthesis
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("sagawa/ReactionT5v2-retrosynthesis", return_tensors="pt")
model = AutoModelForSeq2SeqLM.from_pretrained("sagawa/ReactionT5v2-retrosynthesis")

inp = tokenizer('CCN(CC)CCNC(=S)NC1CCCc2cc(C)cnc21', return_tensors='pt')
output = model.generate(**inp, num_beams=1, num_return_sequences=1, return_dict_in_generate=True, output_scores=True)
output = tokenizer.decode(output['sequences'][0], skip_special_tokens=True).replace(' ', '').rstrip('.')
print(output) # 'CCN(CC)CCN=C=S.Cc1cnc2c(c1)CCCC2N'



