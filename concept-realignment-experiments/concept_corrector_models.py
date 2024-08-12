import torch
from torch import nn


class LSTMConceptCorrector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, input_format='original_and_intervened_inplace'):
        super(LSTMConceptCorrector, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_format = input_format
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    
    def prepare_initial_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device), \
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

        
    def forward(self, inputs, already_intervened_concepts, original_predictions, hidden):
        if self.input_format == 'original_and_intervened_inplace':
            x = already_intervened_concepts * inputs + (1 - already_intervened_concepts) * original_predictions
        elif self.input_format == 'previous_output':
            x = inputs
        else:
            raise Exception(f"Input format {self.input_format} is not supported")

        lstm_out, hid = self.lstm(x, hidden)

        output = torch.sigmoid(self.fc(lstm_out))

        output = already_intervened_concepts * inputs + (1 - already_intervened_concepts) * output

        return output, hid


    def forward_single_timestep(self, inputs, already_intervened_concepts, original_predictions, hidden, input_format='original_and_intervened_inplace'):
        inputs_, already_intervened_concepts_, original_predictions_ = torch.unsqueeze(inputs, dim=1), torch.unsqueeze(already_intervened_concepts, dim=1), torch.unsqueeze(original_predictions, dim=1)

        output, hid = self.forward(inputs_, already_intervened_concepts_, original_predictions_, hidden)
        
        output = torch.squeeze(output, dim=1)

        return output, hid


# Define the neural network
class GRUConceptCorrector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, input_format='original_and_intervened_inplace'):
        super(GRUConceptCorrector, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_format = input_format
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def prepare_initial_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


    def forward(self, inputs, already_intervened_concepts, original_predictions, hidden):
        if self.input_format == 'original_and_intervened_inplace':
            x = already_intervened_concepts * inputs + (1 - already_intervened_concepts) * original_predictions
        elif self.input_format == 'previous_output':
            x = inputs
        else:
            raise Exception(f"Input format {self.input_format} is not supported")

        lstm_out, hid = self.gru(x, hidden)

        output = torch.sigmoid(self.fc(lstm_out))

        output = already_intervened_concepts * inputs + (1 - already_intervened_concepts) * output

        return output, hid


    def forward_single_timestep(self, inputs, already_intervened_concepts, original_predictions, hidden):
        inputs_, already_intervened_concepts_, original_predictions_ = torch.unsqueeze(inputs, dim=1), torch.unsqueeze(already_intervened_concepts, dim=1), torch.unsqueeze(original_predictions, dim=1)

        output, hid = self.forward(inputs_, already_intervened_concepts_, original_predictions_, hidden)
        
        output = torch.squeeze(output, dim=1)

        return output, hid


class RNNConceptCorrector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, input_format='original_and_intervened_inplace'):
        super(RNNConceptCorrector, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_format = input_format
        
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def prepare_initial_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


    def forward(self, inputs, already_intervened_concepts, original_predictions, hidden):
        if self.input_format == 'original_and_intervened_inplace':
            x = already_intervened_concepts * inputs + (1 - already_intervened_concepts) * original_predictions
        elif self.input_format == 'previous_output':
            x = inputs
        else:
            raise Exception(f"Input format {self.input_format} is not supported")

        lstm_out, hid = self.rnn(x, hidden)

        output = torch.sigmoid(self.fc(lstm_out))

        output = already_intervened_concepts * inputs + (1 - already_intervened_concepts) * output

        return output, hid


    def forward_single_timestep(self, inputs, already_intervened_concepts, original_predictions, hidden):
        inputs_, already_intervened_concepts_, original_predictions_ = torch.unsqueeze(inputs, dim=1), torch.unsqueeze(already_intervened_concepts, dim=1), torch.unsqueeze(original_predictions, dim=1)

        output, hid = self.forward(inputs_, already_intervened_concepts_, original_predictions_, hidden)
        
        output = torch.squeeze(output, dim=1)

        return output, hid
        

class NNConceptCorrector(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size, input_format='original_and_intervened_inplace', dropout_rate=0.2):
        super(NNConceptCorrector, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_format = input_format
        self.dropout_rate = dropout_rate
        self.hidden_layers = hidden_layers
        
        # Input layer
        self.layers = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size)])
        self.layers.extend([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.hidden_layers - 1)])

        # Batch Normalization for each hidden layer
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for _ in range(self.hidden_layers)])

        # Dropout for each hidden layer
        self.dropouts = nn.ModuleList([nn.Dropout(p=self.dropout_rate) for _ in range(self.hidden_layers)])

        # Output layer
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

        for _ in range(self.hidden_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    
    def prepare_initial_hidden(self, batch_size, device):
        return None


    def forward_single_timestep(self, x, mask, original_predictions, hidden):
        inputs = x
        
        if self.input_format == 'original_and_intervened_inplace':
            x = x * mask + original_predictions * (1-mask)
        elif self.input_format == 'previous_output':
            pass
        else:
            raise Exception(f"Input format {self.input_format} is not supported")

        for layer, batch_norm, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            x = batch_norm(x)
            x = torch.relu(x)
            x = dropout(x)

        # Last hidden layer without ReLU
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        x = inputs * mask + x * (1-mask)

        if torch.min(x) < 0 or torch.max(x) > 1:
            print("Value of Output of NN outside range")
            print(torch.min(x), torch.max(x))
            print("range of mask:")
            print(torch.min(mask), torch.max(mask))

        return x, None


    def forward(self, x, mask, original_predictions, hidden): # x = [batch_size, seq_length, num_concepts]
        # reshape x to [batch_size x seq_length, num_concepts]
        original_size = x.size()
        x, mask, original_predictions = x.reshape(-1, x.size(-1)), mask.reshape(-1, mask.size(-1)), original_predictions.reshape(-1, original_predictions.size(-1))
        out, _ = self.forward_single_timestep(x, mask, original_predictions, hidden)

        # reshape out back into the original shape
        out = out.reshape(original_size)

        return out, None







# class NNConceptCorrector(nn.Module):
#     def __init__(self, input_size, hidden_size, hidden_layers, output_size, input_format='original_and_intervened_inplace'):
#         super(NNConceptCorrector, self).__init__()

#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.input_format = input_format

#         self.input_layer = nn.Linear(input_size, hidden_size)
#         self.hidden_layers = nn.ModuleList()

#         for _ in range(hidden_layers):
#             self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

#         self.output_layer = nn.Linear(hidden_size, output_size)

    
#     def initial_hidden(self, batch_size):
#         return None


#     def forward_single_timestep(self, x, mask, original_predictions, hidden):
#         inputs = x
        
#         if self.input_format == 'original_and_intervened_inplace':
#             x = x * mask + original_predictions * (1-mask)
#         elif self.input_format == 'previous_output':
#             pass
#         else:
#             raise Exception(f"Input format {self.input_format} is not supported")

#         x = self.input_layer(x)
#         x = torch.relu(x)

#         for hidden_layer in self.hidden_layers:
#             x = hidden_layer(x)
#             x = torch.relu(x)

#         x = self.output_layer(x)
#         x = torch.sigmoid(x)
        
#         x = inputs * mask + x * (1-mask)

#         return x, None


#     def forward(self, x, mask, original_predictions, hidden): # x = [batch_size, seq_length, num_concepts]
#         # reshape x to [batch_size x seq_length, num_concepts]
#         original_size = x.size()
#         x, mask, original_predictions = x.reshape(-1, x.size(-1)), mask.reshape(-1, mask.size(-1)), original_predictions.reshape(-1, original_predictions.size(-1))
#         out, _ = self.forward_single_timestep(x, mask, original_predictions, hidden)

#         # reshape out back into the original shape
#         out = out.reshape(original_size)

#         return out, None