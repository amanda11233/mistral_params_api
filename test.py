import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

output_dir = "Mistral-7-int4-dolly"

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(output_dir)


from datasets import load_dataset
from random import randrange


# Load dataset from the hub and get a sample
dataset = load_dataset("json", data_files={'train':'p2.json'})['train']
sample = dataset[randrange(len(dataset))]

input = "Update auto apply layout station to False, start station to 0, and enable skew girder end section"

prompt = f""" <s>[INST] <<SYS>>\n
From the Instruction below, extract the parameters and generate the output in JSON Response.
Return only the parameters specified in the input. Don't repreat the parameters.
Identify the most similar field name from the following list for the extracted parameters:
[user_label,
bridge_type,
skew_girdir,
vertical_girdir,
bearing,
deck_width,
start_station,
auto_apply_layout_station,
deck_section_type,
span_location,
span_label,
bearing_offset_at_end,
automate_beam_girder_cl_offset,
beam_girder_cl_offset_along_bent_at_start,
start_diaphragm,
beam_girder_cl_offset_along_bent_at_end,
beam_girder_spacing_start,
plan_configuration,
skew_girder_end_section,
vertical_girder_end_section,
auto_adjust_diaphragm_depth,
span_segment_interval,
path_interval]
<</SYS>>
### Input: Update auto apply layout station to False, start station to 0, and enable skew girder end section[/INST] \n  
{{"auto_apply_layout_station" : False, 
"start_station" : 0, 
"skew_girder_end_section" : True}}</s>
<s>[INST]
### Input: Set the vertical grider end section to disable, user label to Multi Span Bridge-1 , skew grinder end section to enable and plan configuration to 0. [/INST]
"""

print(prompt)
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)

#print(f"Prompt:\n{input}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
