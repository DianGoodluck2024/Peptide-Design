from huggingface_hub import login
import random

import torch
import torch.nn.functional as F

from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization import get_model_tokenizers
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer as EsmFunctionTokenizer,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation

# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
login(token="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")
tokenizers = get_model_tokenizers()


model = ESM3_sm_open_v0("cuda")

# PDB 1UTN
#sequence = "M K T F I F L | G S L I N S Q"
sequence = "M|KTFI"
tokens = tokenizers.sequence.encode(sequence)

vocab = tokenizers.sequence.get_vocab()

# Print the total number of tokens
print("Total vocabulary size:", len(vocab))

# Print all tokens and their corresponding IDs
for token, token_id in vocab.items():
    print(f"Token: '{token}', ID: {token_id}")



sequence_tokens = torch.tensor(tokens, dtype=torch.int64)

sequence_tokens = sequence_tokens.cuda().unsqueeze(0)

# output = model.forward(
#     sequence_tokens=sequence_tokens, function_tokens=function_tokens
# )
output = model.forward(
     sequence_tokens=sequence_tokens
 )
#print(output.sequence_logits.shape)
print(output.embeddings.shape)