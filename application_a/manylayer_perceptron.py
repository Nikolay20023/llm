import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):

        super().__init__()

        self.layers = torch.nn.Sequential(

            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            torch.nn.Linear(20, num_outputs)
        )
    
    def forward(self, x):
        logits = self.layers(x)
        return logits
    

if __name__ == "__main__":
    torch.manual_seed(123)

    X = torch.rand((1, 50))

    model = NeuralNetwork(50, 3)

    out = model(X)

    print(out)
