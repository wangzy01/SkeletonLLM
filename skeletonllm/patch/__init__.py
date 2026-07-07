from .pad_data_collator import concat_pad_data_collator
from .train_dataloader_patch import replace_train_dataloader
from .train_sampler_patch import replace_train_sampler


def replace_internlm2_attention_class(*args, **kwargs):
    from .internlm2_packed_training_patch import replace_internlm2_attention_class as fn

    return fn(*args, **kwargs)


def replace_llama_attention_class(*args, **kwargs):
    from .llama_packed_training_patch import replace_llama_attention_class as fn

    return fn(*args, **kwargs)


def replace_llama_rmsnorm_with_fused_rmsnorm(*args, **kwargs):
    from .llama_rmsnorm_monkey_patch import replace_llama_rmsnorm_with_fused_rmsnorm as fn

    return fn(*args, **kwargs)


def replace_phi3_attention_class(*args, **kwargs):
    from .phi3_packed_training_patch import replace_phi3_attention_class as fn

    return fn(*args, **kwargs)


def replace_qwen2_attention_class(*args, **kwargs):
    from .qwen2_packed_training_patch import replace_qwen2_attention_class as fn

    return fn(*args, **kwargs)


def apply_liger_kernel_to_internvit(*args, **kwargs):
    from .internvit_liger_monkey_patch import apply_liger_kernel_to_internvit as fn

    return fn(*args, **kwargs)


__all__ = [
    'concat_pad_data_collator',
    'replace_internlm2_attention_class',
    'replace_llama_attention_class',
    'replace_llama_rmsnorm_with_fused_rmsnorm',
    'replace_phi3_attention_class',
    'replace_qwen2_attention_class',
    'replace_train_dataloader',
    'replace_train_sampler',
    'apply_liger_kernel_to_internvit',
]
