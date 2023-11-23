# import torch
# import torch.nn.functional as F

# from torch import nn



# def create_alignment(base_mat, duration_predictor_output):
#     N, L = duration_predictor_output.shape
#     for i in range(N):
#         count = 0
#         for j in range(L):
#             for k in range(duration_predictor_output[i][j]):
#                 base_mat[i][count+k][j] = 1
#             count = count + duration_predictor_output[i][j]
#     return base_mat

# class LengthRegulator(nn.Module):
#     """Length Regulator"""

#     def __init__(self):
#         super(LengthRegulator, self).__init__()

#     def LR(self, x, duration, max_len):
#         output = list()
#         mel_len = list()
#         for batch, expand_target in zip(x, duration):
#             expanded = self.expand(batch, expand_target)
#             output.append(expanded)
#             mel_len.append(expanded.shape[0])

#         if max_len is not None:
#             output = pad(output, max_len)
#         else:
#             output = pad(output)

#         return output, torch.LongTensor(mel_len).to(x.device)

#     def expand(self, batch, predicted):
#         out = list()

#         for i, vec in enumerate(batch):
#             expand_size = predicted[i].item()
#             out.append(vec.expand(max(int(expand_size), 0), -1))
#         out = torch.cat(out, 0)

#         return out

#     def forward(self, x, duration, max_len):
#         output, mel_len = self.LR(x, duration, max_len)
#         return output, mel_len


# def pad(input_ele, mel_max_length=None):
#     if mel_max_length:
#         max_len = mel_max_length
#     else:
#         max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

#     out_list = list()
#     for i, batch in enumerate(input_ele):
#         if len(batch.shape) == 1:
#             one_batch_padded = F.pad(
#                 batch, (0, max_len - batch.size(0)), "constant", 0.0
#             )
#         elif len(batch.shape) == 2:
#             one_batch_padded = F.pad(
#                 batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
#             )
#         out_list.append(one_batch_padded)
#     out_padded = torch.stack(out_list)
#     return out_padded

# # class LengthRegulator(nn.Module):
# #     """ Length Regulator """

# #     def __init__(self):
# #         super(LengthRegulator, self).__init__()
# #         # self.duration_predictor = DurationPredictor(model_config)

# #     def LR(self, x, duration_predictor_output, mel_max_length=None):
# #         expand_max_len = torch.max(
# #             torch.sum(duration_predictor_output, -1), -1
# #         )[0].long().item()
# #         print(duration_predictor_output.shape, expand_max_len)
# #         alignment = torch.zeros(
# #             duration_predictor_output.size(0),
# #             expand_max_len,
# #             duration_predictor_output.size(1)
# #         ).numpy()
# #         alignment = create_alignment(
# #             alignment,
# #             duration_predictor_output.cpu().numpy()
# #         )
# #         alignment = torch.from_numpy(alignment).to(x.device)

# #         output = alignment @ x
# #         if mel_max_length:
# #             output = F.pad(
# #                 output,
# #                 (0, 0, 0, mel_max_length-output.size(1), 0, 0)
# #             )
# #         return output

# #     # def forward(self, input, durations=None, target=None, mel_max_length=None):
# #     #     if target is not None:
# #     #         output = self.LR(input, target, mel_max_length)
# #     #         return output, durations
        
# #     #     # durations are rounded already
# #     #     output = self.LR(input, durations)
# #     #     mel_pos = torch.arange(1, output.shape[1] + 1, dtype=torch.long).unsqueeze(0)
# #     #     return output, mel_pos.to(output.device)
# #     def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
# #         duration_predictor_output = self.duration_predictor(x)

# #         if target is not None:
# #             output = self.LR(x, target, mel_max_length)
# #             return output, duration_predictor_output
# #         else:
# #             duration_predictor_output = (((torch.exp(duration_predictor_output) - 1) * alpha) + 0.5).int()
# #             duration_predictor_output[duration_predictor_output < 0] = 0

# #             output = self.LR(x, duration_predictor_output)
# #             mel_pos = torch.stack(
# #                 [torch.Tensor([i+1  for i in range(output.size(1))])]
# #             ).long().to(x.device)
# #             return output, mel_pos

# #         # for sample, sample_durations in zip(input, durations):
# #         #     expanded_sample = []
# #         #     for embed, duration in zip(sample, sample_durations):
# #         #         expanded_sample.append(embed.expand(
# #         #             max(int(duration), 0), -1
# #         #         ))
# #         #     output.append(torch.cat(expanded_sample, 0).to(input.device))
        
# #         # mel_lengths = torch.tensor(
# #         #     [sample.shape[0] for sample in output]
# #         # ).long().to(input.device)
        
# #         # if mel_max_length is not None:
# #         #     max_length = mel_max_length
# #         # else:
# #         #     max_length = max(sample.shape[1] for sample in output)
        
# #         # output = torch.stack([
# #         #     F.pad(
# #         #         sample, pad=(0, 0, 0, max_length - sample.shape[1]),
# #         #         mode='constant', value=0.
# #         #     ) for sample in output
# #         # ]).to(input.device)

# #         # return output, mel_lengths



import torch
import torch.nn.functional as F
from torch import nn

def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat

class LengthRegulator(nn.Module):
    """ Length Regulator """
    def __init__(self):
        super(LengthRegulator, self).__init__()
        # self.duration_predictor = DurationPredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1
        )[0].long().item()

        alignment = torch.zeros(
            duration_predictor_output.size(0),
            expand_max_len,
            duration_predictor_output.size(1)
        ).numpy()

        alignment = create_alignment(
            alignment,
            duration_predictor_output.cpu().numpy()
        )
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x

        if mel_max_length:
            output = F.pad(
                output,
                (0, 0, 0, mel_max_length-output.size(1), 0, 0)
            )
        return output

    def forward(self, input, durations=None, target=None, mel_max_length=None):
        if target is not None:
            output = self.LR(input, target, mel_max_length)
            return output, durations

        # durations are rounded already
        output = self.LR(input, durations)
        mel_pos = torch.arange(1, output.shape[1] + 1, dtype=torch.long).unsqueeze(0)
        return output, mel_pos.to(output.device)

