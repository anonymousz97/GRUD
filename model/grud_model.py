import torch 
from torch import nn
from torch._C import dtype
from .gru_model import GRUModel
from torch.nn import functional as F
from .activations import get_activation

class CustomDiscriminator(nn.Module):
    """
    Custom Discriminator with custom_linear is a list of dictionary with input_dim and output_dim (e.g. [{'input_dim': 100, 'output_dim': 50}])
    """
    def __init__(self, custom_linear=[], activation='gelu', custom_linear_activation='gelu', device='cpu'):
        super().__init__()
        assert custom_linear, 'custom_linear should not be empty'
        self.custom_linear = nn.ModuleList([nn.Linear(i['input_dim'], i['output_dim']) for i in custom_linear])
        self.dense_prediction = nn.Linear(custom_linear[-1]['output_dim'], 1)
        self.activation = get_activation(activation)
        self.custom_act = get_activation(custom_linear_activation)
    
    def forward(self, discriminator_hidden_states):
        for i in self.custom_linear:
            discriminator_hidden_states = i(discriminator_hidden_states)
            discriminator_hidden_states = self.custom_act(discriminator_hidden_states)
        logits = torch.squeeze(self.activation(self.dense_prediction(discriminator_hidden_states)), dim=-1)
        return logits

class Discriminator(nn.Module):
    def __init__(self, hidden_size, activation='gelu'):
        super().__init__()
        # self.dense = nn.Linear(hidden_size, 768)
        # self.dense2 = nn.Linear(768, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_prediction = nn.Linear(hidden_size, 1)
        self.activation = get_activation(activation)
    
    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        # hidden_states = self.dense2(hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = torch.squeeze(self.dense_prediction(hidden_states), dim=-1)
        return logits


class GRUD(nn.Module):
    def __init__(self,model_config, device):
        super(GRUD,self).__init__()
        self.num_classes = model_config['num_classes']
        self.generator = GRUModel(model_config, device)
        # self.discriminator = Discriminator(self.num_classes, 'sigmoid')
        self.discriminator = CustomDiscriminator(
            custom_linear = [
                {"input_dim":self.num_classes,"output_dim":self.num_classes * 2},
                {"input_dim":self.num_classes * 2,"output_dim":self.num_classes},
                #{"input_dim":self.num_classes,"output_dim":self.num_classes},
            ],
            activation = 'sigmoid',
            device = device
        )

    def forward(self, input_):
        discriminator_logit = None
        logit_prediction = self.generator(input_)
        fake_logit = torch.stack(logit_prediction)
        logit_generator = torch.cat(logit_prediction, dim=0)
        prediction = F.softmax(fake_logit, dim=-1)
        if self.training:
            discriminator_logit = self.discriminator(fake_logit.detach()).T
        return logit_generator, prediction, discriminator_logit
    



    
