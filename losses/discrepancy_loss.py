import torch

class DiscrepancyLoss(torch.nn.Module):
    def __init__(self, model1, model2):
        super(DiscrepancyLoss, self).__init__()
        self.model1 = model1
        self.model2 = model2
    
    def forward(self):
        param1 = self.collect_params(self.model1)
        param2 = self.collect_params(self.model2)
        loss = (torch.matmul(param1, param2) / (torch.norm(param1) * torch.norm(param2)) + 1)
        return loss

    def collect_params(self, network):
        params = []
        for p in network.parameters():
            params.append(p.view(-1))
        params = torch.cat(params, dim=0)
        return params