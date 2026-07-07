# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

__all__ = ['InternVisionConfig', 'InternVisionModel',
           'InternVLChatConfig', 'InternVLChatModel']


def __getattr__(name):
    if name == 'InternVisionConfig':
        from .configuration_intern_vit import InternVisionConfig

        return InternVisionConfig
    if name == 'InternVLChatConfig':
        from .configuration_internvl_chat import InternVLChatConfig

        return InternVLChatConfig
    if name == 'InternVisionModel':
        from .modeling_intern_vit import InternVisionModel

        return InternVisionModel
    if name == 'InternVLChatModel':
        from .modeling_internvl_chat import InternVLChatModel

        return InternVLChatModel
    raise AttributeError(name)
