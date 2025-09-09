"""
Load model architecture from different sources
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from src.model.dream.configuration_dream import ODreamConfig
from src.model.dream.modeling_dream import DreamForCausalLM

# LLaDA v2
from src.model.llada_v2.configuration_llada import LLaDAV2Config
from src.model.llada_v2.modeling_llada import LLaDAV2ModelLM

from src.model.dream_flash.configuration_dream import ODreamConfig as DreamFlashConfig
from src.model.dream_flash.modeling_dream import DreamForCausalLM as DreamFlashForCausalLM

from peft import PeftModel

# register the Dream model
AutoConfig.register("odream", ODreamConfig)
AutoModelForCausalLM.register(ODreamConfig, DreamForCausalLM)


# register the LLaDA v2 model (Block-Cached)
AutoConfig.register("ollada_v2", LLaDAV2Config) 
AutoModelForCausalLM.register(LLaDAV2Config, LLaDAV2ModelLM)

# register the Dream Flash model
AutoConfig.register("odream_flash", DreamFlashConfig)
AutoModelForCausalLM.register(DreamFlashConfig, DreamFlashForCausalLM)


# TODO: expand this list to support more model architectures
MODEL_LIBRARY_MAP = {
    'meta-llama/Llama-2-7b-chat-hf': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.2-1B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.2-3B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.1-8B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'nvidia/OpenMath2-Llama3.1-8B': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.1-8B': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-2-13b-hf': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-2-7b-hf': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.1-70B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    "lmsys/vicuna-7b-v1.3": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2-7B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-7B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-3B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-Math-1.5B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-1.5B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ('transformers', 'AutoModelForCausalLM'),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": ('transformers', 'AutoModelForCausalLM'),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": ('transformers', 'AutoModelForCausalLM'),

    # Dream models
    "GSAI-ML/LLaDA-8B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Dream-org/Dream-v0-Instruct-7B": ('transformers', 'DreamForCausalLM'),
    "Dream-org/Dream-Flash-Instruct-7B": ('transformers', 'DreamFlashForCausalLM'),

    # LLaDA v2 model
    "GSAI-ML/LLaDA-8B-Instruct-v2": ('transformers', 'LLaDAV2ModelLM'),

    "Qwen/Qwen2.5-7B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-7B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-3B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-Math-1.5B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-1.5B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen3-0.6B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen3-8B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-Math-1.5B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-0.5B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-Omni-3B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-Omni-7B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-Math-1.5B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-Math-7B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-Math-7B-Instruct": ('transformers', 'AutoModelForCausalLM'),

    # Add missing Qwen models
    "Qwen/Qwen2.5-0.5B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-1.5B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-3B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-14B": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-14B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-Math-PRM-7B": ('transformers', 'AutoModelForCausalLM'),

    # Llama 3.2 models
    "meta-llama/Llama-3.2-8B": ('transformers', 'AutoModelForCausalLM'),
    "meta-llama/Llama-3.2-8B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    "meta-llama/Llama-3.2-1B": ('transformers', 'AutoModelForCausalLM'),
    "meta-llama/Llama-3.2-3B": ('transformers', 'AutoModelForCausalLM'),

    "meta-llama/Meta-Llama-3-8B": ('transformers', 'AutoModelForCausalLM'),
    "meta-llama/Meta-Llama-3-8B-Instruct": ('transformers', 'AutoModelForCausalLM'),


    # DeepSeek-R1-Distill-Qwen-7B
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ('transformers', 'AutoModelForCausalLM'),
}

def load_and_fuse_peft_model(model_name, ckpt):
    base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True)

    peft_model = PeftModel.from_pretrained(base_model, ckpt)
    peft_model.merge_and_unload()
    return peft_model


class ModelMap:
    def __init__(self, model_name:str, pretrained=None, peft=False):
        self.model_name = model_name
        self.pretrained = pretrained
        self.peft = False
        
    def fetch(self, device_map="auto"):
        print(f"Attempting to fetch model: '{self.model_name}'")
        print(f"Available models: {list(MODEL_LIBRARY_MAP.keys())}")
        if self.model_name not in MODEL_LIBRARY_MAP:
            raise ValueError(f"Model: {self.model_name} is unknown! Available models: {MODEL_LIBRARY_MAP.keys()}")

        lib_name, sub_name = MODEL_LIBRARY_MAP[self.model_name]

        if lib_name == "transformers":
            if self.pretrained is None:
                if not self.peft:
                    if "llama" in self.model_name or "Llama" in self.model_name:
                        config = OLlamaConfig.from_pretrained(self.model_name)
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            config=config,
                            load_in_8bit=False,
                            torch_dtype=torch.float16,
                            device_map=device_map,
                            trust_remote_code=True,
                        )
                    # Dream 7b variants
                    elif "Dream" in self.model_name or "dream" in self.model_name:
                        # remove "-Custom" from the model name if it exists
                        if "-v2" in self.model_name:
                            self.model_name = self.model_name.replace("-v2", "-v0")
                            config = DreamV2Config().from_pretrained(self.model_name)
                        elif "-Flash" in self.model_name:
                            self.model_name = self.model_name.replace("-Flash", "-v0")
                            config = DreamFlashConfig().from_pretrained(self.model_name)
                        else:
                            config = ODreamConfig().from_pretrained(self.model_name)
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            config=config,
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                        )
                    # LLaDA variants
                    elif "LLaDA" in self.model_name or "llada" in self.model_name:
                        if "-v2" in self.model_name:
                            self.model_name = self.model_name.replace("-v2", "")
                            config = LLaDAV2Config().from_pretrained(self.model_name)
                            model = LLaDAV2ModelLM.from_pretrained(
                                self.model_name,
                                config=config,
                                trust_remote_code=True,
                                torch_dtype=torch.float16,
                            )
                            return model
                        else:
                            # config = LLaDAConfig().from_pretrained(self.model_name)
                            raise ValueError(f"LLaDA model: {self.model_name} is not supported yet!")

                        # import pdb; pdb.set_trace()

                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            config=config,
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                        )

                        import pdb; pdb.set_trace()

                    # Qwen2.5-Omni variants
                    # elif "Qwen2.5-Omni" in self.model_name:
                    #     config = Qwen2_5OmniConfig.from_pretrained(self.model_name)
                    #     model = Qwen2_5OmniForCausalLM.from_pretrained(
                    #         self.model_name,
                    #         config=config,
                    #         trust_remote_code=True,
                    #         torch_dtype=torch.float16,
                    #     )
                    #     return model
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            load_in_8bit=False,
                            torch_dtype=torch.float16,
                            # device_map="auto",
                            device_map=device_map,
                            trust_remote_code=True,
                        )
                else:
                    print(f"Prepraring PEFT model for {self.model_name}")
                    model = load_and_fuse_peft_model(self.model_name, ckpt=self.pretrained)
            else:
                print(f"Loading model from: {self.pretrained}")
                model = AutoModelForCausalLM.from_pretrained(self.pretrained)

        else:
            raise ValueError(f"Unknown model library {lib_name}")

        return model