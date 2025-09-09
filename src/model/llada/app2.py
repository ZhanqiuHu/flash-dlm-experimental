import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import time
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, 
                                  torch_dtype=torch.bfloat16).to(device)

# Constants
MASK_TOKEN = "[MASK]"
MASK_ID = 126336  # The token ID of [MASK] in LLaDA

def parse_constraints(constraints_text):
    """Parse constraints in format: 'position:word, position:word, ...'"""
    constraints = {}
    if not constraints_text:
        return constraints
        
    parts = constraints_text.split(',')
    for part in parts:
        if ':' not in part:
            continue
        pos_str, word = part.split(':', 1)
        try:
            pos = int(pos_str.strip())
            word = word.strip()
            if word and pos >= 0:
                constraints[pos] = word
        except ValueError:
            continue
    
    return constraints

def format_chat_history(history):
    """
    Format chat history for the LLaDA model.
    Args:
        history: List of [user_message, assistant_message] pairs.
    Returns:
        List of dictionaries with keys "role" and "content".
    """
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Skip if None (for the latest user message)
            messages.append({"role": "assistant", "content": assistant_msg})
    return messages

def add_gumbel_noise(logits, temperature):
    '''
    Apply Gumbel noise for sampling from a categorical distribution.
    If temperature <= 0, returns logits unchanged.
    '''
    if temperature <= 0:
        return logits
        
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    Precompute the number of tokens to update (transfer) at each denoising step,
    based on the mask indices in a block and the total number of steps.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def generate_response_with_visualization(model, tokenizer, device, messages, gen_length=64, steps=32, 
                                         constraints=None, temperature=0.0, cfg_scale=0.0, block_length=32,
                                         remasking='low_confidence'):
    """
    Generate text with LLaDA model using the diffusion denoising process.
    Returns a list of visualization states (one per denoising step) and the final text.
    
    This function implements the same sampling as in generate.py.
    """
    # Process constraints: convert word constraints to token IDs.
    if constraints is None:
        constraints = {}
    processed_constraints = {}
    for pos, word in constraints.items():
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, token_id in enumerate(tokens):
            processed_constraints[pos + i] = token_id

    # Prepare prompt using chat template
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(chat_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    prompt_length = input_ids.shape[1]
    
    # Initialize the sequence: prompt followed by gen_length masks.
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
    x[:, :prompt_length] = input_ids.clone()
    
    # Initialize visualization states for the response region.
    visualization_states = []
    initial_state = [(MASK_TOKEN, "#444444") for _ in range(gen_length)]
    visualization_states.append(initial_state)
    
    # Apply constraints (if any) to the response region.
    for pos, token_id in processed_constraints.items():
        absolute_pos = prompt_length + pos
        if absolute_pos < x.shape[1]:
            x[:, absolute_pos] = token_id
    
    # Mark prompt positions to keep them unchanged.
    prompt_index = (x != MASK_ID)
    
    # Ensure block_length is within gen_length.
    if block_length > gen_length:
        block_length = gen_length
    
    # Calculate number of blocks.
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1
    
    # Adjust steps per block.
    steps_per_block = steps // num_blocks
    if steps_per_block < 1:
        steps_per_block = 1

    # For each block, run denoising steps.
    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])
        
        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        # If the block is already complete, skip it.
        if not block_mask_index.any():
            continue
        
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == MASK_ID)
            if not mask_index.any():
                break

            # Run model inference, applying CFG if enabled.
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = MASK_ID
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            
            # Sample candidate tokens via Gumbel noise.
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Compute confidence scores.
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
            
            # Only update tokens within the current block.
            x0_p[:, block_end:] = -float('inf')
            
            # For positions that are not masked, keep the current token.
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))
            
            # Select tokens to update based on top-k confidence.
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                block_confidence = confidence[j, block_start:block_end]
                if i < steps_per_block - 1:
                    k_val = min(num_transfer_tokens[j, i].item(), block_confidence.numel())
                    _, select_indices = torch.topk(block_confidence, k=k_val)
                    select_indices = select_indices + block_start
                    transfer_index[j, select_indices] = True
                else:
                    transfer_index[j, block_start:block_end] = mask_index[j, block_start:block_end]
            
            # Update the sequence with the selected tokens.
            x = torch.where(transfer_index, x0, x)
            
            # Reapply constraints to ensure they are not overwritten.
            for pos, token_id in processed_constraints.items():
                absolute_pos = prompt_length + pos
                if absolute_pos < x.shape[1]:
                    x[:, absolute_pos] = token_id
            
            # Build a visualization state for the response region.
            current_state = []
            for pos in range(gen_length):
                absolute_pos = prompt_length + pos
                if x[0, absolute_pos] == MASK_ID:
                    current_state.append((MASK_TOKEN, "#444444"))
                else:
                    # Color coding: newly revealed tokens (if they changed in this step) get a distinct color.
                    token = tokenizer.decode([x[0, absolute_pos].item()], skip_special_tokens=True)
                    # Use a simple scheme: light green for newly updated tokens, light blue otherwise.
                    if transfer_index[0, absolute_pos]:
                        current_state.append((token, "#66CC66"))
                    else:
                        current_state.append((token, "#6699CC"))
            visualization_states.append(current_state)
    
    # Decode the final generated response.
    response_tokens = x[0, prompt_length:]
    final_text = tokenizer.decode(response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return visualization_states, final_text

# New function for manual stepping
def get_next_visualization(vis_states, current_index):
    """
    Given a list of visualization states and the current index,
    return the next visualization state and the updated index.
    If at the end, return the last state.
    """
    if current_index < len(vis_states) - 1:
        current_index += 1
    return vis_states[current_index], current_index

css = '''
.category-legend{display:none}
button{height: 60px}
'''

def create_chatbot_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# LLaDA - Large Language Diffusion Model Demo")
        gr.Markdown("[model](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), [project page](https://ml-gsai.github.io/LLaDA-demo/)")
        
        # STATE MANAGEMENT
        chat_history = gr.State([])
        vis_states_state = gr.State([])  # Will store the list of visualization states
        vis_index_state = gr.State(0)    # Current index in the vis_states list
        
        # UI COMPONENTS
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Conversation", height=500)
                with gr.Group():
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="Your Message", 
                            placeholder="Type your message here...",
                            show_label=False
                        )
                        send_btn = gr.Button("Send")
                
                constraints_input = gr.Textbox(
                    label="Word Constraints", 
                    info="Place words at specific positions using 'position:word' format. E.g., '0:Once, 5:upon, 10:time'",
                    placeholder="0:Once, 5:upon, 10:time",
                    value=""
                )
            with gr.Column(scale=2):
                output_vis = gr.HighlightedText(
                    label="Denoising Process Visualization",
                    combine_adjacent=False,
                    show_legend=True,
                )
        
        # Advanced generation settings
        with gr.Accordion("Advanced Generation Settings", open=False):
            with gr.Row():
                gen_length_slider = gr.Slider(
                    minimum=16, maximum=128, value=64, step=8,
                    label="Generation Length (tokens)"
                )
                steps_slider = gr.Slider(
                    minimum=8, maximum=64, value=32, step=4,
                    label="Total Denoising Steps"
                )
            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                    label="Temperature"
                )
                cfg_scale_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                    label="CFG Scale"
                )
            with gr.Row():
                block_length_slider = gr.Slider(
                    minimum=8, maximum=128, value=32, step=8,
                    label="Block Length"
                )
                remasking_radio = gr.Radio(
                    choices=["low_confidence", "random"],
                    value="low_confidence",
                    label="Remasking Strategy"
                )
            with gr.Row():
                pause_checkbox = gr.Checkbox(label="Pause at each step (manual stepping)", value=False)
                visualization_delay_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.1,
                    label="Visualization Delay (seconds)"
                )
        
        # Hidden states for response and visualization stepping
        current_response = gr.Textbox(
            label="Current Response",
            placeholder="The assistant's response will appear here...",
            lines=3,
            visible=False
        )
        
        # Next Step button for manual stepping (only active when pause is enabled)
        next_step_btn = gr.Button("Next Step", visible=False)
        
        # Clear button
        clear_btn = gr.Button("Clear Conversation")
        
        # HELPER FUNCTIONS
        def add_message(history, message, response):
            """Append a new message pair to the conversation history."""
            history = history.copy()
            history.append([message, response])
            return history
            
        def user_message_submitted(message, history):
            """Process a submitted user message."""
            if not message.strip():
                return history, history, ""
            history = add_message(history, message, None)
            history_for_display = history.copy()
            return history, history_for_display, ""
            
        def bot_response(history, gen_length, steps, constraints, temperature, cfg_scale, block_length, remasking, pause):
            """
            Generate bot response with visualization.
            If pause is True, the full list of visualization states is returned and the first state is shown.
            Otherwise, states are streamed automatically.
            """
            if not history:
                return history, [], ""
            
            last_user_message = history[-1][0]
            messages = format_chat_history(history[:-1])
            messages.append({"role": "user", "content": last_user_message})
            parsed_constraints = parse_constraints(constraints)
            
            vis_states, response_text = generate_response_with_visualization(
                model, tokenizer, device, 
                messages, 
                gen_length=gen_length, 
                steps=steps,
                constraints=parsed_constraints,
                temperature=temperature,
                cfg_scale=cfg_scale,
                block_length=block_length,
                remasking=remasking
            )
            
            history[-1][1] = response_text
            
            if pause:
                # Store vis_states and reset index
                return history, vis_states[0], response_text, vis_states, 0
            else:
                # Automatic mode: stream states with delay
                # Here we simply return the final state after streaming (for simplicity)
                # In a real streaming setup, you could yield intermediate states.
                for state in vis_states[1:]:
                    time.sleep(visualization_delay_slider.value)
                return history, vis_states[-1], response_text, [], 0
        
        def next_step(vis_states, vis_index):
            """Return the next visualization state and update the index."""
            if not vis_states:
                return gr.update(value=""), vis_index
            state, new_index = get_next_visualization(vis_states, vis_index)
            return state, new_index
        
        def clear_conversation():
            """Clear conversation history and reset visualization states."""
            return [], [], "", [], 0
        
        # EVENT HANDLERS
        
        # Clear button handler
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chat_history, chatbot_ui, current_response, vis_states_state, vis_index_state]
        )
        
        # When user submits a message (either via textbox submit or send button)
        msg_submit = user_input.submit(
            fn=user_message_submitted,
            inputs=[user_input, chat_history],
            outputs=[chat_history, chatbot_ui, user_input]
        )
        send_click = send_btn.click(
            fn=user_message_submitted,
            inputs=[user_input, chat_history],
            outputs=[chat_history, chatbot_ui, user_input]
        )
        
        # After adding the message, generate the bot response.
        # Note: the outputs now include extra states if pause mode is enabled.
        msg_submit.then(
            fn=bot_response,
            inputs=[
                chat_history, gen_length_slider, steps_slider, constraints_input, 
                temperature_slider, cfg_scale_slider, block_length_slider, remasking_radio, pause_checkbox
            ],
            outputs=[chat_history, chatbot_ui, current_response, vis_states_state, vis_index_state]
        )
        send_click.then(
            fn=bot_response,
            inputs=[
                chat_history, gen_length_slider, steps_slider, constraints_input, 
                temperature_slider, cfg_scale_slider, block_length_slider, remasking_radio, pause_checkbox
            ],
            outputs=[chat_history, chatbot_ui, current_response, vis_states_state, vis_index_state]
        )
        
        # Show/hide the Next Step button based on pause mode.
        pause_checkbox.change(
            lambda pause: gr.update(visible=pause),
            inputs=[pause_checkbox],
            outputs=[next_step_btn]
        )
        
        # Next Step button: when clicked, show next visualization state.
        next_step_btn.click(
            fn=next_step,
            inputs=[vis_states_state, vis_index_state],
            outputs=[output_vis, vis_index_state]
        )
        
    return demo

# Launch the demo
if __name__ == "__main__":
    demo = create_chatbot_demo()
    demo.queue().launch(share=True)
