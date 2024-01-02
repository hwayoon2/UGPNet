from torch import nn
from torch.nn import functional as F
import math
import random
import torch
from basicsr.archs.stylegan2_arch import StyleGAN2Generator

class StyleGAN2GeneratorFChurch(StyleGAN2Generator):

    def __init__(self,
                 out_size=256,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 narrow=1,
                 basecode_size=16):
        super(StyleGAN2GeneratorFChurch, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
            narrow=narrow)
        self.basecode_size=basecode_size

    def forward(self,
                styles,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False,
                basecode=None,
                check_basecode=False,
                input_is_basecode=False,
                return_f=False,
                scales=None,
                biases=None,
                half_sft=False,
                sft_idx=None):
        """Forward function for StyleGAN2GeneratorSFT.
        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        if not input_is_basecode:
            # style codes -> latents with Style MLP layer
            if not input_is_latent:
                styles = [self.style_mlp(s) for s in styles]
            # style truncation
            if truncation < 1:
                style_truncation = []
                for style in styles:
                    style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
                styles = style_truncation
            # get style latents with injection
            if len(styles) == 1:
                inject_index = self.num_latent

                if styles[0].ndim < 3:
                    # repeat latent code for all the layers
                    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                else:  # used for encoder with different latent code for each layer
                    latent = styles[0]
            elif len(styles) == 2:  # mixing noises
                if inject_index is None:
                    inject_index = random.randint(1, self.num_latent - 1)
                latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
                latent = torch.cat([latent1, latent2], 1)

        else:
            latent = styles


        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        features = {}
        i = 1
        idx = 0
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)

            if out.shape[2] == self.basecode_size:
                if check_basecode:
                    return out
                elif input_is_basecode:
                    out = basecode.contiguous()
                    out = conv2(out, latent[:, i + 1], noise=noise2)
                    skip = to_rgb(out, latent[:, i + 2], None)                   
                else:
                    out = conv2(out, latent[:, i + 1], noise=noise2)
                    skip = to_rgb(out, latent[:, i + 2], skip)

            elif out.shape[2] == 256:
                out = conv2(out, latent[:, i + 1], noise=noise2)
                f = out
                skip = to_rgb(out, latent[:, i + 2], skip)

            else:
                out = conv2(out, latent[:, i + 1], noise=noise2)
                skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_f:
            return image, f
        else:
            return image



class StyleGAN2GeneratorF(StyleGAN2Generator):

    def __init__(self,
                 out_size=512,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 narrow=1,
                 basecode_size=16):
        super(StyleGAN2GeneratorF, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
            narrow=narrow)
        self.basecode_size=basecode_size

    def forward(self,
                styles,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False,
                basecode=None,
                check_basecode=False,
                input_is_basecode=False,
                return_f=False,
                scales=None,
                biases=None,
                half_sft=False,
                sft_idx=None):
        """Forward function for StyleGAN2GeneratorSFT.
        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        if not input_is_basecode:
            # style codes -> latents with Style MLP layer
            if not input_is_latent:
                styles = [self.style_mlp(s) for s in styles]
            # style truncation
            if truncation < 1:
                style_truncation = []
                for style in styles:
                    style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
                styles = style_truncation
            # get style latents with injection
            if len(styles) == 1:
                inject_index = self.num_latent

                if styles[0].ndim < 3:
                    # repeat latent code for all the layers
                    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                else:  # used for encoder with different latent code for each layer
                    latent = styles[0]
            elif len(styles) == 2:  # mixing noises
                if inject_index is None:
                    inject_index = random.randint(1, self.num_latent - 1)
                latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
                latent = torch.cat([latent1, latent2], 1)

        else:
            latent = styles


        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        features = {}
        i = 1
        idx = 0
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)

            if out.shape[2] == self.basecode_size:
                if check_basecode:
                    return out
                elif input_is_basecode:
                    out = basecode.contiguous()
                    out = conv2(out, latent[:, i + 1], noise=noise2)
                    skip = to_rgb(out, latent[:, i + 2], None)                   
                else:
                    out = conv2(out, latent[:, i + 1], noise=noise2)
                    skip = to_rgb(out, latent[:, i + 2], skip)

            elif out.shape[2] == 512:
                out = conv2(out, latent[:, i + 1], noise=noise2)
                f = out
                skip = to_rgb(out, latent[:, i + 2], skip)

            else:
                out = conv2(out, latent[:, i + 1], noise=noise2)
                skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_f:
            return image, f
        else:
            return image