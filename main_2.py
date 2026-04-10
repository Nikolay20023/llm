import torch
import tiktoken

from gelu import FeedForward, GELU
import torch.nn as nn
from shortcut_conn import ExampleDeepNeuralNetwork, print_gradients
from gptmodel import GPTModel
from text_encode_decode import generate_text_simple, token_ids_to_text


def main():

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
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


    batch.append(torch.tensor(tokinizer.encode(txt1)))
    batch.append(torch.tensor(tokinizer.encode(txt2)))

    

    batch = torch.stack(batch, dim=0)

    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)


    # print("Input batch:\n", batch)
    # print("\nOutput shape:", out.shape)
    # print(out)

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

    
    # layer_sizes = [3, 3, 3, 3, 3, 1]
    # sample_input = torch.tensor([[1., 0., -1.]])
    # torch.manual_seed(123)
    # model_without_shortcut = ExampleDeepNeuralNetwork(
    #     layer_sizes, use_shortcut=False
    # )

    # torch.manual_seed(123)
    # model_with_shortcut = ExampleDeepNeuralNetwork(
    #     layer_sizes, use_shortcut=True
    # )

    # print_gradients(model_with_shortcut, sample_input)
    # torch.manual_seed(123)
    # x = torch.rand(2, 4, 768)
    # block = TransformerBlock(GPT_CONFIG_124M)
    # output = block(x)

    # print_gradients(model_without_shortcut, sample_input)
    # print("Input shape:", x.shape)
    # print("Output shape:", output.shape) 

    # start_context = "Hello , i am"

    # encoded = tokinizer.encode(start_context)
    # print("encoded", encoded)
    # encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()

    # out = generate_text_simple(
    #     model=model,
    #     idx=encoded_tensor,
    #     max_mew_token=6,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )

    # print("Output:", out)
    # print("Output length:", len(out[0]))

    # decode_text = tokinizer.decode(out.squeeze(0).tolist())
    # print("Decode text:", decode_text)
    inputs = torch.tensor([[16833, 3626, 6100],
                           [40, 1107, 588]])

    targets = torch.tensor([[3626, 6100, 345],
                            [1107, 588, 11311]])
    
    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    # print("Token IDs: \n, ", token_ids)

    # print(f"Targets batch 1 :, {token_ids_to_text(targets[0], tokinizer)}")
    # print(f"Outputs batch 1 :"
    #       f"{token_ids_to_text(token_ids[0].flatten(), tokinizer)}")
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))

    avg_log_probas = torch.mean(log_probas)

    neg_avg_log_probas = avg_log_probas * -1

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)

if __name__ == "__main__":
    main()