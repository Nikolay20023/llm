import torch
import tiktoken

from gptmodel import DummyGPTModel, LayerNorm
from gelu import FeedForward, GELU
import torch.nn as nn
from shortcut_conn import ExampleDeepNeuralNetwork


def main():

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1025,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    tokinizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every days holds a"


    # batch.append(torch.tensor(tokinizer.encode(txt1)))
    # batch.append(torch.tensor(tokinizer.encode(txt2)))

    # batch = torch.stack(batch, dim=0)

    # torch.manual_seed(123)

    # model = DummyGPTModel(GPT_CONFIG_124M)

    # logits = model(batch)
    # print("Output shape:", logits.shape)
    # print(logits)

    # torch.manual_seed(123)
    # batch_example = torch.randn(2, 5)
    # layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())

    # out = layer(batch_example)


    # ln = LayerNorm(emb_dim=5)
    # out_ln = ln(batch_example)

    # mean = out_ln.mean(dim=-1, keepdim=True)
    # var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    # torch.set_printoptions(sci_mode=False)
    
    # ffn = FeedForward(GPT_CONFIG_124M)
    # x = torch.rand(2, 3, 768)
    # out = ffn(x)

    
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )

if __name__ == "__main__":
    main()