import torch


class flow_model_torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        #t = torch.linspace(0,1,500)
        return self.model(t, x)
