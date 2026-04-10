import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_embedding_size, target_embedding_size):
        return self.decoder(self.encoder(input_embedding_size), target_embedding_size)



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_layer = nn.Linear(input_size, hidden_size)
        
    def forward(self, input):
        output = self.encoder_layer(input)
        return output


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.decoder_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output = self.decoder_layer(input)
        return output

embedding = nn.Embedding(input_size, hidden_size)