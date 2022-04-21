from lib2to3.pgen2.tokenize import tokenize
import transformers
from transformers import BertTokenizer    #bm
import torch
import poptorch
import pdb   #bm

# A bert model from hugging face. See the packaged BERT example for actual usage.
pretrained_weights = 'mrm8488/bert-medium-finetuned-squadv2'


# For later versions of transformers, we need to wrap the model and set
# return_dict to False
class WrappedModel(torch.nn.Module):
 def __init__(self):
   super().__init__()
   self.wrapped = transformers.BertForQuestionAnswering.from_pretrained(pretrained_weights)

 def forward(self, input_ids, attention_mask, token_type_ids):
   return self.wrapped.forward(input_ids, attention_mask, token_type_ids, return_dict=False)

 def __getattr__(self, attr):
   try:
     return torch.nn.Module.__getattr__(self, attr)
   except AttributeError:
     return getattr(self.wrapped, attr)

# pdb.set_trace()
model = WrappedModel()

# A handy way of seeing the names of all the layers in the network.
print(model)

# All layers before "model.bert.encoder.layer[0]" will be on IPU 0 and all layers from
# "model.bert.encoder.layer[0]" onwards (inclusive) will be on IPU 1.
model.bert.encoder.layer[0] = poptorch.BeginBlock(model.bert.encoder.layer[0], ipu_id=1)

# Now all layers before layer are on IPU 1 and this layer onward is on IPU 2
model.bert.encoder.layer[2] = poptorch.BeginBlock(model.bert.encoder.layer[2], ipu_id=2)

# Finally all layers from this layer till the end of the network are on IPU 3.
model.bert.encoder.layer[4] = poptorch.BeginBlock(model.bert.encoder.layer[4], ipu_id=3)

# We must batch the data by at least the number of IPUs. Each IPU will still execute
# whatever the model batch size is.
data_batch_size = 4
# data_batch_size = 1     #bm

opts = poptorch.Options()
opts.deviceIterations(data_batch_size)
# opts.Training.gradientAccumulation(4)

#bm
ipumodel = poptorch.inferenceModel(model=model, options=opts)
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
# inputs = tokenizer("hello, my cat is cute", return_tensors='pt')
inputs = tokenizer(["hello, my cat is cute","hi hi hi","by by by","ga ga ga"], return_tensors='pt', padding=True)
outputs = model(**inputs)
pdb.set_trace()
outputs2 = ipumodel(**inputs)

pdb.set_trace()
print('done')
