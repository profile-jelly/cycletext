import torch
import torch.nn as nn
from peft.tuners.lora import LoraLayer

class CycleAwareLoraLinear(nn.Linear, LoraLayer):
    def __init__(self, *args, **kwargs):
        nn.Linear.__init__(self, *args)
        LoraLayer.__init__(self, **kwargs)
        self.lora_scale = None

    def set_lora_scale(self, scale):
        self.lora_scale = scale

    def forward(self, x):
        result = nn.Linear.forward(self, x)

        if self.r > 0:
            lora_out = self.lora_B(self.lora_A(x)) * self.scaling

            if hasattr(self, "lora_scale") and self.lora_scale is not None:
                scale = self.lora_scale.detach()
                while scale.dim() < lora_out.dim():
                    scale = scale.unsqueeze(-1)
                lora_out = lora_out * scale

            result = result + lora_out

        return result
