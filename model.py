import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from box import Box

num2type = {
    1: 'transformer',
    2: 'lstm',
    3: 'linear'
}


class RecognitionModel(nn.Module):
    def __init__(self, encoder_type=2, decoder_type=2, num_classes=6, device='cuda', **kwargs):
        super(RecognitionModel, self).__init__()
        self.encoder_type = num2type[encoder_type]
        self.decoder_type = num2type[decoder_type]
        self.device = device
        kwargs = Box(kwargs)

        if self.encoder_type == 'lstm':
            self.kinematic_encoder = nn.LSTM(kwargs.kin_encoder_input_dim, kwargs.kin_encoder_hidden_dim, batch_first=True,
                                             num_layers=kwargs.kin_encoder_num_layers, bidirectional=True)
            self.video_encoder = nn.LSTM(kwargs.vid_encoder_input_dim, kwargs.vid_encoder_hidden_dim, batch_first=True,
                                         num_layers=kwargs.vid_encoder_num_layers, bidirectional=True)
            if kwargs.include_video == 2:
                dim_factor = 2
                self.aux_video_encoder = nn.LSTM(kwargs.vid_encoder_input_dim, kwargs.vid_encoder_hidden_dim, batch_first=True,
                                                 num_layers=kwargs.vid_encoder_num_layers, bidirectional=True)
            else:
                dim_factor = 1
                self.aux_video_encoder = None
            self.dropout = nn.Dropout(kwargs.dropout)
        else:
            raise NotImplementedError(f"Encoder type {self.encoder_type} is not supported.")

        decoder_input_dim = 2 * (kwargs.kin_encoder_hidden_dim + dim_factor * kwargs.vid_encoder_hidden_dim) if self.encoder_type == 'lstm'\
            else (kwargs.kin_encoder_input_dim + dim_factor * kwargs.vid_encoder_input_dim)

        if self.decoder_type == 'lstm':
            self.decoder = nn.LSTM(decoder_input_dim, kwargs.decoder_hidden_dim, batch_first=True, num_layers=kwargs.decoder_num_layers,
                                   bidirectional=True)
        elif self.decoder_type == 'linear':
            self.decoder = nn.Identity()
        else:
            raise NotImplementedError(f"Decoder type {self.decoder_type} is not supported.")

        in_features = 2 * kwargs.decoder_hidden_dim if self.decoder_type == "lstm" else decoder_input_dim
        self.linear = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, kin_in, vid_in, mask):
        kin_in = kin_in.permute(0, 2, 1)
        #encoders
        kin_out = self._lstm_forward(kin_in, mask, type='kinematic')
        vid_out = self._lstm_forward(vid_in[0], mask, type='video')
        if len(vid_in) == 2:
            aux_vid_out = self._lstm_forward(vid_in[1], mask, type='aux_video')
            features = torch.cat([kin_out, aux_vid_out, vid_out], dim=2)
        else:
            features = torch.cat([kin_out, vid_out], dim=2)
        # decoder
        decoding = self._lstm_forward(features, mask, type='combined') if self.decoder_type != "linear" else self.decoder(features)
        # linear
        output = self.linear(decoding)
        return [output.permute(0, 2, 1)]

    def _lstm_forward(self, input, mask, type):
        """
        Forward path of the LSTM model
        :param input: input to LSTM
        :param mask:
        :param type: which LSTM is used
        :return: output from LSTM
        """
        lengths = torch.sum(mask, dim=-1).to(dtype=torch.int64).to(device='cpu')
        input = self.dropout(input)
        packed_input = pack_padded_sequence(input, lengths=lengths, batch_first=True, enforce_sorted=False)
        if type == 'kinematic':
            output, _ = self.kinematic_encoder(packed_input)
        elif type == 'video':
            output, _ = self.video_encoder(packed_input)
        elif type == 'aux_video':
            output, _ = self.aux_video_encoder(packed_input)
        elif type == 'combined':
            output, _ = self.decoder(packed_input)
        else:
            raise NotImplementedError(f"Features type '{type}' is not supported.")
        unpacked_out, unpacked_out_lengths = pad_packed_sequence(output, padding_value=-1, batch_first=True)
        unpacked_out = self.dropout(unpacked_out)
        return unpacked_out
