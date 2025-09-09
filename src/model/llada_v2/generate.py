import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate_baseline(
    model, 
    prompt, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0.,         
    cfg_scale=0., 
    remasking='low_confidence', 
    mask_id=126336, 
    # NEW: early stopping
    early_stop=False, 
    eos_token_id=None):
    '''
    Baseline implementation of diffusion generation without block caching.
    
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        early_stop: Whether to enable early stopping when EOS token is detected or all tokens are unmasked.
    '''     
    if early_stop and eos_token_id is None:
        raise ValueError("eos_token_id must be provided when early_stop is True")

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start: block_end] == mask_id)
        
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)

            global_step_idx = num_block * block_length + i
            
            # Early stop if all tokens are unmasked
            if early_stop and not mask_index.any():
                # print in blue
                BLUE = '\033[94m'
                RESET = '\033[0m'
                print(f"{BLUE}Early stopping at global step {global_step_idx}{RESET}")
                return x  # It's okay to return when all tokens are unmasked
            
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf  # Only consider tokens in current and previous blocks

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            # Early stop if EOS token is detected
            if early_stop:
                eos_mask = (x == eos_token_id)
                if eos_mask.any():
                    # Find the first EOS token position
                    BLUE = '\033[94m'
                    RESET = '\033[0m'
                    # print(f"{BLUE}Early stopping at global step {global_step_idx}{RESET}")
                    eos_pos = eos_mask.nonzero()[:, 1].min()
                    # Mask everything after EOS with EOS token
                    x[:, eos_pos+1:] = eos_token_id
                    # Continue processing the current block instead of returning

    return x

@torch.no_grad()
def generate_block_cached(model, 
    prompt, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0.,
    cfg_scale=0., 
    remasking='low_confidence', 
    mask_id=126336,

    # Add early stopping
    early_stop=False,
    eos_token_id=None,
             
    # this functionuses block caching
    # NEW: block caching parameters
    use_full_query_attn=False):
    '''
    Block-cached implementation of diffusion generation.
    '''

    if early_stop and eos_token_id is None:
        raise ValueError("eos_token_id must be provided when early_stop is True")

    prompt_len = prompt.shape[1]
    gen_len = gen_length

    # initialize the output tensor
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # block caching
    for num_block in range(num_blocks):
        # Original code:
        # Note: mask_index is the indices of masked tokens in the current block
        # block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        # num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        # New code with block boundaries:
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start: block_end] == mask_id)
        
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        
        for i in range(steps):

            # global step index
            global_step_idx = num_block * block_length + i

            mask_index = (x == mask_id)

            # Early stop if all tokens are unmasked
            if early_stop and not mask_index.any():
                BLUE = '\033[94m'
                RESET = '\033[0m'
                print(f"{BLUE}Early stopping at global step {global_step_idx}{RESET}")
                return x  # It's okay to return when all tokens are unmasked
            
            # Determine save_cache and clean_idx for this step
            save_cache = (i == steps - 1)  # Save cache at the last step of each block
            clean_idx = block_start  # Clean tokens up to the start of current block
            
            if cfg_scale > 0.:
                raise NotImplementedError("CFG scale is not implemented for block caching")
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                # Original code:
                # logits = model(x_).logits
                # New code with block caching:
                logits = model(x_, 
                                # NEW: block caching parameters
                                use_block_caching=True,
                                use_full_query_attn=use_full_query_attn,
                                block_size=block_length,  # Use block_length as block_size
                                save_cache=save_cache,
                                clean_idx=clean_idx
                                ).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # Original code:
                # logits = model(x).logits
                # x shape: b, l
                
                # Calling forward pass of the model
                # With block caching, pass in the full x at first global step

                if global_step_idx == 0:
                    # TODO: Reset/Initialize the block cache
                    bsz = x.shape[0]
                    L = x.shape[1]  # Total sequence length including prompt
                    print(f"Resetting cache: bsz={bsz}, L={L}, block_size={block_length}")
                    
                    # Initialize block cache for each layer
                    if hasattr(model.model.transformer, 'blocks'):
                        for block in model.model.transformer.blocks:
                            block.reset_block_cache(bsz, L, block_length)
                    else:  # block_groups
                        for block_group in model.model.transformer.block_groups:
                            for block in block_group:
                                block.reset_block_cache(bsz, L, block_length)
                    model.model.cache_initialized = True

                    # import pdb; pdb.set_trace()

                    # New code with block caching:
                    logits = model(x,
                                 # NEW: block caching parameters
                                 use_block_caching=True,
                                 use_full_query_attn=use_full_query_attn,
                                 block_size=block_length,  # Use block_length as block_size
                                 save_cache=True,
                                 clean_idx=block_start,
                                 ).logits
                else:
                    # New code with block caching:
                    # feed in only the current block and future blocks
                    # print(f"Feeding in from: {block_start}")
                    x_len = x.shape[1]
                    logits = model(x[:, block_start:],
                                 # NEW: block caching parameters
                                 use_block_caching=True,
                                 use_full_query_attn=use_full_query_attn,
                                 block_size=block_length,  # Use block_length as block_size
                                 save_cache=save_cache, # only save cache at the last step of each block
                                 clean_idx=block_end, # save up to the end of the current block
                                 ).logits

                    # check output logic
                    # print(f"Output shape: {logits.shape}")

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            # x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            x0_slice = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p_slice = F.softmax(logits_with_noise.to(torch.float64), dim=-1)
                x0_p_slice = torch.squeeze(
                    torch.gather(p_slice, dim=-1, index=torch.unsqueeze(x0_slice, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p_slice = torch.rand((x0_slice.shape[0], x0_slice.shape[1]), device=x0_slice.device)
            else:
                raise NotImplementedError(remasking)

            if x0_slice.shape[1] < x.shape[1]:
                # Pad left with the previous tokens
                pad_left = x[:, :block_start]
                x0 = torch.cat([pad_left, x0_slice], dim=1)
                pad_conf = torch.full(
                    (1, block_start), 
                    -np.inf, 
                    device=x.device, 
                    dtype=x0_p_slice.dtype)
                x0_p = torch.cat([pad_conf, x0_p_slice], dim=1) # (1, L_full)
            else:
                x0, x0_p = x0_slice, x0_p_slice

            # Original code:
            # x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            
            # New code with block boundaries:
            x0_p[:, block_end:] = -np.inf  # Only consider tokens in current and previous blocks

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            # Early stop if EOS token is detected
            if early_stop:
                eos_mask = (x == eos_token_id)
                if eos_mask.any():
                    # Find the first EOS token position
                    BLUE = '\033[94m'
                    RESET = '\033[0m'
                    # print(f"{BLUE}Early stopping at global step {global_step_idx}{RESET}")
                    eos_pos = eos_mask.nonzero()[:, 1].min()
                    # Mask everything after EOS with EOS token
                    x[:, eos_pos+1:] = eos_token_id
                    # Continue processing the current block instead of returning

    return x

@ torch.no_grad()
def generate(
    model, 
    prompt, 
    steps=128,
    gen_length=128, block_length=128, temperature=0.,
    cfg_scale=0., remasking='low_confidence', mask_id=126336,
             
    # NEW: block caching parameters
    use_block_caching=False,
    use_full_query_attn=False,

    # NEW: early stopping
    early_stop=False,
    eos_token_id=None,

    ):
    '''
    Main generation function that can switch between baseline and block-cached implementations.
    
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        # NEW: block caching parameters
        use_block_caching: Whether to use block caching
        use_full_query_attn: Whether to use full sequence for query attention
        early_stop: Whether to enable early stopping when EOS token is detected or all tokens are unmasked.
        eos_token_id: The token ID for EOS token. Required if early_stop is True.
    '''

    if early_stop and eos_token_id is None:
        raise ValueError("eos_token_id must be provided when early_stop is True")

    if use_block_caching:
        out = generate_block_cached(model, prompt, steps, gen_length, block_length, temperature,
                           cfg_scale, remasking, mask_id,
                           use_full_query_attn=use_full_query_attn,
                           # NEW: early stopping
                           early_stop=early_stop,
                           eos_token_id=eos_token_id)
    else:
        out = generate_baseline(model, prompt, steps, gen_length, block_length, temperature,
                           cfg_scale, remasking, mask_id,
                           # NEW: early stopping
                           early_stop=early_stop,
                           eos_token_id=eos_token_id)
    return out


# Original implementation with block caching (commented out)
# @ torch.no_grad()
# def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
#              cfg_scale=0., remasking='low_confidence', mask_id=126336,
#              # NEW: block caching parameters
#              use_block_caching=False,
#              use_full_query_attn=False):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#         # NEW: block caching parameters
#         use_block_caching: Whether to use block caching
#         use_full_query_attn: Whether to use full sequence for query attention
#     '''
#     x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()

#     prompt_index = (x != mask_id)

#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length

#     assert steps % num_blocks == 0
#     steps = steps // num_blocks

#     for num_block in range(num_blocks):
#         # Original code:
#         # Note: mask_index is the indices of masked tokens in the current block
#         # block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
#         # num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
#         # New code with block boundaries:
#         block_start = prompt.shape[1] + num_block * block_length
#         block_end = prompt.shape[1] + (num_block + 1) * block_length
#         block_mask_index = (x[:, block_start: block_end] == mask_id)
        
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        
#         for i in range(steps):
#             mask_index = (x == mask_id)
            
#             # Determine save_cache and clean_idx for this step
#             save_cache = (i == steps - 1)  # Save cache at the last step of each block
#             clean_idx = block_start  # Clean tokens up to the start of current block
            
#             if cfg_scale > 0.:
#                 un_x = x.clone()
#                 un_x[prompt_index] = mask_id
#                 x_ = torch.cat([x, un_x], dim=0)
#                 # Original code:
#                 # logits = model(x_).logits
#                 # New code with block caching:
#                 if use_block_caching:
#                     logits = model(x_, 
#                                  # NEW: block caching parameters
#                                  use_block_caching=use_block_caching,
#                                  use_full_query_attn=use_full_query_attn,
#                                  block_size=block_length,  # Use block_length as block_size
#                                  save_cache=save_cache,
#                                  clean_idx=clean_idx
#                                  ).logits
#                 logits, un_logits = torch.chunk(logits, 2, dim=0)
#                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#             else:
#                 # Original code:
#                 # logits = model(x).logits
#                 # x shape: b, l
                
#                 # New code with block caching:
#                 if use_block_caching:
#                     logits = model(x,
#                                  # NEW: block caching parameters
#                                  use_block_caching=use_block_caching,
#                                  use_full_query_attn=use_full_query_attn,
#                                  block_size=block_length,  # Use block_length as block_size
#                                  save_cache=save_cache,
#                                  clean_idx=clean_idx).logits
#                 else:
#                     logits = model(x).logits
#                     # x shape: b, l

#             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#             x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

#             if remasking == 'low_confidence':
#                 p = F.softmax(logits.to(torch.float64), dim=-1)
#                 x0_p = torch.squeeze(
#                     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#             elif remasking == 'random':
#                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#             else:
#                 raise NotImplementedError(remasking)

#             # Original code:
#             # x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            
#             # New code with block boundaries:
#             x0_p[:, block_end:] = -np.inf  # Only consider tokens in current and previous blocks

#             x0 = torch.where(mask_index, x0, x)
#             confidence = torch.where(mask_index, x0_p, -np.inf)

#             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#             for j in range(confidence.shape[0]):
#                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                 transfer_index[j, select_index] = True
#             x[transfer_index] = x0[transfer_index]

#     return x

# Example usage
# def main():
#     device = 'cuda'

#     model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
#     tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

#     prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

#     # Add special tokens for the Instruct model. The Base model does not require the following two lines.
#     m = [{"role": "user", "content": prompt}, ]
#     prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

#     input_ids = tokenizer(prompt)['input_ids']
#     input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

#     out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
#     print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


# if __name__ == '__main__':
#     main()