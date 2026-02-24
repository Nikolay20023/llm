import re
import tiktoken
import torch


from tokenizator import SimpleTokenizatoV1, SimpleTokenizerV2
from torch.utils.data import Dataset, DataLoader
from dataset import GPTDatasetv1
from selfattention import SelfAteentionV2, SelfAteention, CausalAttention, MultiHeadAttentionWrapper, MultiHeadAttention


def main():


    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(list(set(preprocessed)))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_words)


    vocab = {token: integer for integer, token in enumerate(all_words)}

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " ".join((text1, text2))

    tokenizer = SimpleTokenizerV2(vocab=vocab)

    print(tokenizer.decode(tokenizer.encode(text=text)))


def _main_tiktoken():
    tokinizer = tiktoken.get_encoding('gpt2')

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    enc_text = tokinizer.encode(raw_text)

    enc_sample = enc_text[50:]

    context_size = 4

    x = enc_sample[: context_size]
    y = enc_sample[1: context_size + 1]
    
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokinizer.decode(context), "----->", tokinizer.decode([desired]))


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                      stride=128, shuffle=True, drop_last=True,
                      num_workers=0):
    tokinizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetv1(txt, tokinizer, max_length, stride)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def softmax_native(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
    

if __name__ == "__main__":

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # tokinizer = tiktoken.get_encoding('gpt2')

    # enc_text = tokinizer.encode(raw_text)

    # enc_sample = enc_text[50:]

    # context_size = 4
 
    # x = enc_sample[:context_size]
    # y = enc_sample[1: context_size + 1]

    # for i in range(1, context_size + 1):
    #     context = enc_sample[:i]
    #     desired = enc_sample[i]
    #     print(tokinizer.decode(context), "---->", tokinizer.decode([desired]))

    # dataloader = create_dataloader_v1(
    #     txt=raw_text,
    #     batch_size=8,
    #     max_length=4,
    #     stride=4,
    #     shuffle=False
    # )

    # data_iter = iter(dataloader)
    # inputs, targets = next(data_iter)
    # print(inputs, "\n", targets)
    # inputs, targets = next(data_iter)
    # print(inputs, "\n", targets)

    # vocab_size = 50257
    # outputdim = 256
    # token_emerding_layer = torch.nn.Embedding(vocab_size, outputdim)

    # max_length = 4

    # dataloader = create_dataloader_v1(
    #     raw_text,
    #     batch_size=8,
    #     max_length=max_length,
    #     stride=max_length,
    #     shuffle=False
    # )

    # data_iter = iter(dataloader)

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your 
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts 
        [0.22, 0.58, 0.33], # with 
        [0.77, 0.25, 0.10], # one 
        [0.05, 0.80, 0.55]] # step (x^1)
    )

    d_in = inputs.shape[1]
    d_out = 2

    

    # sa_v1 = SelfAteention(d_in, d_out)

    # sa_v2 = SelfAteentionV2(d_in=d_in, d_out=d_out)


    # queries = sa_v2.W_query(inputs)
    # keys = sa_v2.W_key(inputs)  
    # attn_scores = queries @ keys.T

    # attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    # print(attn_weights)

    # context_lengath = attn_scores.shape[0]

    # mask_simple = torch.tril(torch.ones(context_lengath, context_lengath))
    # print(mask_simple)

    # masked_simple = attn_weights * mask_simple
    # print(masked_simple)

    # rows_sum = masked_simple.sum(dim=-1, keepdim=True)

    # mask = torch.triu(torch.ones(context_lengath, context_lengath), diagonal=1)

    # masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

    # attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)

    # print(attn_weights)

    # torch.manual_seed(123)
    # dropout = torch.nn.Dropout(0.5)
    # example  = torch.ones(6, 6)

    # print(dropout(attn_weights))

    # batch = torch.stack((inputs, inputs), dim=0)

    # context_length = batch.shape[1]
    # ca = CausalAttention(d_in, d_out, context_length, 0.0)

    # context_vecs = ca(batch)

    # print("context_vecs.shape", context_vecs.shape)

    # torch.manual_seed(123)

    # context_length_1 =batch.shape[1]

    # d_in, d_out = 3, 2
    # mha = MultiHeadAttentionWrapper(
    #     d_in, d_out, context_length_1, 0.0, num_heads=2
    # )

    # context_vecs = mha(batch)

    # torch.manual_seed(123)

    # batch_size, context_length, d_in = batch.shape
    # d_out = 2
    # mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    # context_vecs = mha(batch)
    # print(context_vecs)
    # print("context_vecs.shape:", context_vecs.shape)


    # print(batch.shape)

#     query = inputs[1]

#     attn_scores_2 = torch.empty(inputs.shape[0])
#     for i, x_i in enumerate(inputs):
#         attn_scores_2[i] = torch.dot(x_i, query)

    
#     attn_weight_2_tmp = attn_scores_2 / attn_scores_2.sum()

#     # print(attn_weight_2_tmp)
#     # print(attn_weight_2_tmp.sum())

#     attn_weght_2 = torch.softmax(attn_scores_2, dim=0)
# # 
#     # print(attn_weght_2)

#     attn_scores = torch.empty(6, 6)

#     query = inputs[1]

#     context_vec_2 = torch.zeros(query.shape)

#     for i, x_i in enumerate(inputs):
#         context_vec_2 += attn_weght_2[i] * x_i


#     attn_scores = torch.empty(6, 6)

#     for i, x_i in enumerate(inputs):
#         for j, x_j in enumerate(inputs):
#             attn_scores[i, j] = torch.dot(x_i, x_j)


#     attn_scores = inputs @ inputs.T

#     attn_weigths = torch.softmax(attn_scores, dim=-1)


#     context_vector = attn_weigths @ inputs 


#     # Начало мезанизмов самовнимания 

#     x_2 = inputs[1]
#     d_in = inputs.shape[1]
#     d_out = 2

#     torch.manual_seed(123)

#     W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
#     W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
#     W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

#     query_2 = x_2 @ W_query
#     key_2 = x_2 @ W_key
#     value_2 = x_2 @ W_value


#     keys = inputs @ W_key
#     values = inputs @ W_value

#     key_2 = keys[1]

#     attn_scores_22 = query_2.dot(key_2)


#     attn_scores_2 = query_2 @ keys.T


#     d_k = keys.shape[-1]

#     attn_weight_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)

#     value_2 = values[1]

#     context_vec_2 = attn_weight_2 @ values

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1025,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
