import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

import copy
import numpy as np
import math

from eval import segment_bars_with_confidence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape

        assert c1 == c2

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6) # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention)
        attention = attention * padding_mask
        attention = attention.permute(0,2,1)
        out = torch.bmm(proj_val, attention)
        return out, attention

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type): # r1 = r2
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)

        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder','decoder']

        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()


    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2* (self.bl //2)))
        for i in range(self.bl):
            window_mask[:, i, i:i+self.bl] = 1
        return window_mask.to(device)

    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder

        query = self.query_conv(x1)
        key = self.key_conv(x1)

        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)


    def _normal_self_att(self,q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _block_wise_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()

        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb,1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)

        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()


        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)

        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)

        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k, torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v, torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask, torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)

        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # special case when self.bl = 1
        v = torch.cat([v[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0)
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat([padding_mask[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask

        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
#         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out


class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

# class ConvFeedForward2(nn.Module):
#     def __init__(self, dilation, in_channels, out_channels):
#         super(ConvFeedForward2, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         return self.layer(x)

class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.layer(x)


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

# class AttModule2(nn.Module):                #解码器
#     def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
#         super(AttModule2, self).__init__()
#         self.feed_forward = ConvFeedForward2(dilation, in_channels, out_channels)
#         self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
#         self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type,
#                                   stage=stage)  # dilation
#         self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
#         self.dropout = nn.Dropout()
#         self.alpha = alpha
#     def forward(self,x,f,mask):
#         out = self.feed_forward(x)
#         out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
#         out = self.conv_1x1(out)
#         out = self.dropout(out)
#         return (x + out) * mask[:, 0:1, :]

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1) # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)
#         self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in # 2**i
             range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()#         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature

class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='sliding_att',alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att', alpha=exponential_descrease(s))) for s in range(num_decoders)]) # num_decoders


    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature* mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim_RGB, input_dim_FLOW, num_classes,
                 channel_masking_rate):
        self.models = nn.ModuleList([
            MyTransformer(3, num_layers, r1, r2, num_f_maps, input_dim_RGB, num_classes, channel_masking_rate),
            MyTransformer(3, num_layers, r1, r2, num_f_maps, input_dim_FLOW, num_classes, channel_masking_rate)
        ])
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

        for i, model in enumerate(self.models):
            print(f'Model_{["RGB", "FLOW"][i]} Size: {sum(p.numel() for p in model.parameters())}')

        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, batch_gen_tst=None):
        for model in self.models:
            model.train()
            model.to(device)

        optimizers = [
            optim.Adam(self.models[0].parameters(), lr=learning_rate, weight_decay=1e-5),
            optim.Adam(self.models[1].parameters(), lr=learning_rate, weight_decay=1e-5)
        ]

        schedulers = [
            optim.lr_scheduler.ReduceLROnPlateau(optimizers[0], mode='min', factor=0.5, patience=3, verbose=True),
            optim.lr_scheduler.ReduceLROnPlateau(optimizers[1], mode='min', factor=0.5, patience=3, verbose=True)
        ]

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = [0, 0]
            total = [0, 0]

            while batch_gen.has_next():
                batch_input_RGB, batch_input_FLOW, batch_target_RGB, batch_target_FLOW, mask_RGB, mask_FLOW, vids = batch_gen.next_batch(
                    batch_size, False)
                inputs = [t.to(device) for t in [batch_input_RGB, batch_input_FLOW]]
                targets = [t.to(device) for t in [batch_target_RGB, batch_target_FLOW]]
                masks = [t.to(device) for t in [mask_RGB, mask_FLOW]]

                stream_outputs = []
                for i in range(2):  # 0: RGB, 1: FLOW
                    optimizers[i].zero_grad()
                    ps = self.models[i](inputs[i], masks[i])
                    stream_outputs.append(ps)

                loss_total = 0
                for i, ps in enumerate(stream_outputs):
                    loss_stream = 0
                    for p in ps:
                        loss_stream += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                               targets[i].view(-1))
                        loss_stream += 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                     F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * masks[i][:, :, 1:]
                                                         )
                    loss_total += loss_stream

                loss_total.backward()
                epoch_loss += loss_total.item()

                for i in range(2):
                    optimizers[i].step()

                for i, ps in enumerate(stream_outputs):
                    _, predicted = torch.max(ps[-1], 1)
                    correct[i] += ((predicted == targets[i]).float() * masks[i][:, 0, :].squeeze(1)).sum().item()
                    total[i] += torch.sum(masks[i][:, 0, :]).item()

            for scheduler in schedulers:
                scheduler.step(epoch_loss)

            batch_gen.reset()
            print(
                f"[epoch {epoch + 1}]: RGB -- loss = {epoch_loss / len(batch_gen.list_of_examples):.4f}, acc = {correct[0] / total[0]:.4f}")
            print(
                f"[epoch {epoch + 1}]: FLOW -- loss = {epoch_loss / len(batch_gen.list_of_examples):.4f}, acc = {correct[1] / total[1]:.4f}")

            if batch_gen_tst is not None:
                self.test(batch_gen_tst, epoch)
                for i, name in enumerate(["RGB", "FLOW"]):
                    torch.save(self.models[i].state_dict(), f"{save_dir}/epoch-{epoch + 1}{name}.model")
                    torch.save(optimizers[i].state_dict(), f"{save_dir}/epoch-{epoch + 1}{name}.opt")

    def test(self, batch_gen, epoch):
        pass

    def test(self, batch_gen_tst, epoch):
        for model in self.models:
            model.eval()

        correct = [0, 0, 0]
        total = [0, 0, 0]
        if_warp = False

        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input_RGB, batch_input_FLOW, batch_target_RGB, batch_target_FLOW, mask_RGB, mask_FLOW, vids = batch_gen_tst.next_batch(
                    1, if_warp)
                inputs = [t.to(device) for t in [batch_input_RGB, batch_input_FLOW]]
                targets = [batch_target_RGB.to(device), batch_target_FLOW.to(device)]
                masks = [mask_RGB.to(device), mask_FLOW.to(device)]

                outputs = []
                for i in range(2):  # 0:RGB, 1:FLOW
                    p = self.models[i](inputs[i], masks[i])
                    outputs.append(p)

                weight_RGB = 0.5
                weight_FLOW = 0.5
                p_fusion = weight_RGB * outputs[0] + weight_FLOW * outputs[1]

                for i in range(2):  # RGB和FLOW
                    _, predicted = torch.max(outputs[i].data[-1], 1)
                    correct[i] += ((predicted == targets[0]).float() * masks[i][:, 0, :].squeeze(1)).sum().item()
                    total[i] += torch.sum(masks[i][:, 0, :]).item()

                _, predicted_fusion = torch.max(p_fusion.data[-1], 1)
                correct[2] += ((predicted_fusion == targets[0]).float() * masks[0][:, 0, :].squeeze(1)).sum().item()
                total[2] += torch.sum(masks[0][:, 0, :]).item()

        acc_names = ["RGB", "FLOW", "Fusion"]
        for i in range(3):
            acc = correct[i] / total[i] if total[i] > 0 else 0
            print(f"---[epoch {epoch + 1}]---: {acc_names[i]}-- tst acc = {acc:.4f}")

        for model in self.models:
            model.train()
        batch_gen_tst.reset()

    def predict(self, model_dir, results_dir, features_path_RGB, features_path_FLOW, batch_gen_tst, epoch, actions_dict,
                sample_rate):
        for model in self.models:
            model.eval()
            model.to(device)

        model_names = ["RGB", "FLOW"]
        for i, name in enumerate(model_names):
            model_path = f"{model_dir}/epoch-{epoch}{name}.model"
            self.models[i].load_state_dict(torch.load(model_path))
        import time
        with torch.no_grad():
            batch_gen_tst.reset()
            time_start = time.time()

            while batch_gen_tst.has_next():
                batch_input_RGB, batch_input_FLOW, batch_target_RGB, batch_target_FLOW, mask_RGB, mask_FLOW, vids = batch_gen_tst.next_batch(
                    1)
                vid = vids[0]

                features_RGB = np.load(f"{features_path_RGB}{vid.split('.')[0]}.npy").T
                features_FLOW = np.load(f"{features_path_FLOW}{vid.split('.')[0]}.npy")
                features_RGB = features_RGB[:, ::sample_rate]
                features_FLOW = features_FLOW[:, ::sample_rate]

                inputs = [
                    torch.tensor(features_RGB, dtype=torch.float).unsqueeze(0).to(device),
                    torch.tensor(features_FLOW, dtype=torch.float).unsqueeze(0).to(device)
                ]

                predictions = []
                for i in range(2):  # 0:RGB, 1:FLOW
                    pred = self.models[i](inputs[i], torch.ones(inputs[i].size(), device=device))
                    predictions.append(pred)

                weight = [0.5, 0.5]  # RGB和FLOW的融合权重
                predictions_fusion = weight[0] * predictions[0] + weight[1] * predictions[1]

                for i in range(len(predictions_fusion)):
                    confidence, predicted = torch.max(F.softmax(predictions_fusion[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
                    batch_target = batch_target_RGB.squeeze()

                    segment_bars_with_confidence(
                        f"{results_dir}/{vid}_stage{i}.png",
                        confidence.tolist(),
                        batch_target.tolist(),
                        predicted.tolist()
                    )

                    recognition = []
                    for j in range(len(predicted)):
                        action_idx = predicted[j].item()
                        action_label = list(actions_dict.keys())[list(actions_dict.values()).index(action_idx)]
                        recognition.extend([action_label] * sample_rate)

                    f_name = vid.split('/')[-1].split('.')[0]
                    with open(f"{results_dir}/{f_name}", "w") as f_ptr:
                        f_ptr.write("### Frame level recognition: ###\n")
                        f_ptr.write(' '.join(recognition))

            time_end = time.time()



if __name__ == '__main__':
    pass
