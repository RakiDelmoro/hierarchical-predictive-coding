import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.gate_conv_z = nn.Conv2d(channels * 2, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.gate_conv_r = nn.Conv2d(channels * 2, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.candidate_conv = nn.Conv2d(channels * 2, channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, hidden_state, input_data):
        hidden_expanded = hidden_state.expand_as(input_data)
        gates_input = torch.cat([hidden_expanded, input_data], dim=1)
        update_gate = torch.sigmoid(self.gate_conv_z(gates_input))
        reset_gate = torch.sigmoid(self.gate_conv_r(gates_input))
        candidate_input = torch.cat([reset_gate * hidden_expanded, input_data], dim=1)
        candidate_hidden = torch.tanh(self.candidate_conv(candidate_input))
        new_hidden = (1 - update_gate) * hidden_expanded + update_gate * candidate_hidden
        return new_hidden


class TransitionModel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gru_cell = ConvGRU(channels)

    def forward(self, hidden_state, action=None):
        if action is not None:
            combined_input = hidden_state + action
            return self.gru_cell(hidden_state, combined_input)
        return self.gru_cell(hidden_state, hidden_state)
