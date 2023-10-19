#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
# Portions of this code are derived from VisualVoice (CC-BY-NC).
# Original work available at: https://github.com/facebookresearch/VisualVoice
#

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class NVASNet(nn.Module):
    def __init__(self, num_receivers, use_visual=False):
        super(NVASNet, self).__init__()

        self.num_receivers = num_receivers
        self.use_visual = use_visual

        if self.use_visual:
            self.rgb_net = VisualNet(torchvision.models.resnet18(pretrained=True), 3)
            self.depth_net = VisualNet(torchvision.models.resnet18(pretrained=True), 1)

            concat_size = 512 * 2
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = create_conv(concat_size, 512, 1, 0)

        input_channel = 2 * num_receivers
        output_channel = 2

        self.audio_net = AudioNet(64, input_channel, output_channel, self.use_visual)
        self.audio_net.apply(weights_init)

    def forward(self, inputs, disable_detection=False):
        visual_features = []
        if self.use_visual:
            visual_features.append(self.rgb_net(inputs['rgb']))
            visual_features.append(self.depth_net(inputs['depth']))

        if len(visual_features) != 0:
            # concatenate channel-wise
            concat_visual_features = torch.cat(visual_features, dim=1)
            concat_visual_features = self.conv1x1(concat_visual_features)
            concat_visual_features = self.pooling(concat_visual_features)
        else:
            concat_visual_features = None

        # Dereverber
        pred_stft, audio_feat, source_detection = self.audio_net(inputs['input_stft'], concat_visual_features, disable_detection)
        output = {'pred_stft': pred_stft}

        # Source identifier
        output['source_detection'] = source_detection

        if len(visual_features) != 0:
            audio_embed = self.pooling(audio_feat).squeeze(-1).squeeze(-1)
            visual_embed = concat_visual_features.squeeze(-1).squeeze(-1)
            output['audio_feat'] = F.normalize(audio_embed, p=2, dim=1)
            output['visual_feat'] = F.normalize(visual_embed, p=2, dim=1)

        return output


def unet_conv(input_nc, output_nc, use_norm=False, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    if use_norm:
        downnorm = norm_layer(output_nc)
        return nn.Sequential(*[downconv, downnorm, downrelu])
    else:
        return nn.Sequential(*[downconv, downrelu])


def unet_upconv(input_nc, output_nc, outermost=False, use_sigmoid=False, use_tanh=False, use_norm=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    if use_norm and not outermost:
        upnorm = norm_layer(output_nc)
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        if outermost:
            if use_sigmoid:
                return nn.Sequential(*[upconv, nn.Sigmoid()])
            elif use_tanh:
                return nn.Sequential(*[upconv, nn.Tanh()])
            else:
                return nn.Sequential(*[upconv])
        else:
            return nn.Sequential(*[upconv, uprelu])


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, use_norm=False):
        super(conv_block, self).__init__()
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, outermost=False, use_norm=False, scale_factor=(2., 1.)):
        super(up_conv, self).__init__()
        if not outermost:
            if use_norm:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor),
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True)
                )
            else:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor),
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True)
                )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.up(x)
        return x


def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=False, use_relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))
    if use_relu:
        model.append(nn.ReLU())
    return nn.Sequential(*model)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class VisualNet(nn.Module):
    def __init__(self, original_resnet, num_channel=3):
        super(VisualNet, self).__init__()
        original_resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers)  # features before conv1x1

    def forward(self, x):
        x = self.feature_extraction(x)
        return x


class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2, use_visual=False, use_norm=True, audioVisual_feature_dim=512):
        super(AudioNet, self).__init__()

        self.use_visual = use_visual
        if use_visual:
            audioVisual_feature_dim += 512

        # initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf, use_norm=False)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2, use_norm=use_norm)
        self.audionet_convlayer3 = conv_block(ngf * 2, ngf * 4, use_norm=use_norm)
        self.audionet_convlayer4 = conv_block(ngf * 4, ngf * 8, use_norm=use_norm)
        self.audionet_convlayer5 = conv_block(ngf * 8, ngf * 8, use_norm=use_norm)
        self.audionet_convlayer6 = conv_block(ngf * 8, ngf * 8, use_norm=use_norm)
        self.audionet_convlayer7 = conv_block(ngf * 8, ngf * 8, use_norm=use_norm)
        self.audionet_convlayer8 = conv_block(ngf * 8, ngf * 8, use_norm=use_norm)
        self.frequency_time_pool = nn.MaxPool2d([2, 2])
        self.frequency_pool = nn.MaxPool2d([2, 1])
        self.audionet_upconvlayer1 = up_conv(audioVisual_feature_dim, ngf * 8, use_norm=use_norm)
        self.audionet_upconvlayer2 = up_conv(ngf * 16, ngf * 8, use_norm=use_norm)
        self.audionet_upconvlayer3 = up_conv(ngf * 16, ngf * 8, use_norm=use_norm, scale_factor=(2., 2.))
        self.audionet_upconvlayer4 = up_conv(ngf * 16, ngf * 8, use_norm=use_norm, scale_factor=(2., 2.))
        self.audionet_upconvlayer5 = up_conv(ngf * 16, ngf * 4, use_norm=use_norm, scale_factor=(2., 2.))
        self.audionet_upconvlayer6 = up_conv(ngf * 8, ngf * 2, use_norm=use_norm, scale_factor=(2., 2.))
        self.audionet_upconvlayer7 = unet_upconv(ngf * 4, ngf, use_norm=use_norm)
        self.audionet_upconvlayer8 = unet_upconv(ngf * 2, output_nc, True, use_norm=use_norm)
        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

        # Source identifier
        self.source_detector = nn.Sequential(
            nn.Conv2d(audioVisual_feature_dim, 64, kernel_size=3, stride=1, padding=1),  # maintains spatial dimensions as 4x4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # reduces spatial dimensions to 2x2
            nn.Conv2d(64, 32, kernel_size=2, stride=1),  # reduces spatial dimensions to 1x1
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, audio_mix_stft, visual_feat, disable_detection=False):
        audio_conv1feature = self.audionet_convlayer1(audio_mix_stft)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv3feature = self.frequency_time_pool(audio_conv3feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv4feature = self.frequency_time_pool(audio_conv4feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv5feature = self.frequency_time_pool(audio_conv5feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv6feature = self.frequency_time_pool(audio_conv6feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_conv7feature = self.frequency_pool(audio_conv7feature)
        audio_conv8feature = self.audionet_convlayer8(audio_conv7feature)
        audio_conv8feature = self.frequency_pool(audio_conv8feature)

        audioVisual_feature = audio_conv8feature
        if self.use_visual:
            visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1)  # flatten visual feature
            visual_feat = visual_feat.repeat(1, 1, audio_conv8feature.shape[-2],
                                             audio_conv8feature.shape[-1])  # tile visual feature

            audioVisual_feature = torch.cat((visual_feat, audioVisual_feature), dim=1)

        if not disable_detection:
            source_detection = self.source_detector(audioVisual_feature)
        else:
            source_detection = torch.tensor([[0.0]])

        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv7feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv6feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv5feature), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv4feature), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv3feature), dim=1))
        audio_upconv7feature = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv2feature), dim=1))
        prediction = self.audionet_upconvlayer8(torch.cat((audio_upconv7feature, audio_conv1feature), dim=1))

        pred_stft = prediction

        return pred_stft, audio_conv8feature, source_detection
