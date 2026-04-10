import torch
import tiktoken

from gelu import FeedForward, GELU
import torch.nn as nn
from shortcut_conn import ExampleDeepNeuralNetwork, print_gradients
from gptmodel import GPTModel
from text_encode_decode import generate_text_simple, token_ids_to_text, text_to_token_ids, generate
from main import create_dataloader_v1


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    
    elif num_batches is None:
        num_batches = len(data_loader)
    
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evalute_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    
    model.train()

    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model, idx=encoded,
            max_mew_token=50, context_size=context_size
        )
    
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_loses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evalute_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_loses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f}"
                      f"Val loss {val_loss:.3f}")
        
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_loses, val_losses, track_tokens_seen


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


def main():

    tokinizer = tiktoken.get_encoding("gpt2")
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    file_path = "the-verdict.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        txt=train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        txt=val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    torch.manual_seed(123)

    model = GPTModel(
       GPT_CONFIG_124M 
    )

    device = "cpu"

    model.to("cpu")
    model.eval()

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=0.0004, weight_decay=0.1
    # )
    num_epochs = 10

    # train_losses, val_losses, tokens_seen = train_model_simple(
    #     model, train_loader, val_loader, optimizer, device,
    #     num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    #     start_context="Every effort moves you", tokenizer=tokinizer
    # )
    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=text_to_token_ids("Every effort moves you", tokinizer),
    #     max_mew_token=25,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )


    # inverse_vocab = {v: k for k, v in vocab.items()}
    # next_token_logits = torch.tensor(
    #     [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    # )

    # probas = torch.softmax(next_token_logits, dim=0)
    # next_token_id = torch.multinomial(probas, num_samples=1).item()
    # print(inverse_vocab[next_token_id])

    # def print_sampled_tokens(probas):
    #     torch.manual_seed(123)
    #     sample = [torch.multinomial(probas, num_samples=1).item()
    #             for i in range(1_000)]
    #     sampled_ids = torch.bincount(torch.tensor(sample))
    #     for i, freq in enumerate(sampled_ids):
    #         print(f"{freq} x {inverse_vocab[i]}")

    top_k = 3
    # top_lohits, top_pos = torch.topk(next_token_logits, top_k)
    # print("Top logits:", top_lohits)
    # print("Top positions:", top_pos)

    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokinizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output:", token_ids_to_text(token_ids, tokinizer))


if __name__ == "__main__":
    main()