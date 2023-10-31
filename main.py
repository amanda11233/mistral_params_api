from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates

from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging  
 
from pydantic import BaseModel  

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "Mistral-7-int4-dolly"

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


class Instructions(BaseModel):
    instruction: str

 
app = FastAPI()

# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")



logging.basicConfig(level=logging.INFO, filename="./logs/log2.log", filemode="w")

logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
 
# documents = vdb.load_docs_as_loaders()
 
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://csi-mistral.azurewebsites.net",
    "http://csi-mistral.azurewebsites.net",
    "https://csimistral.azurewebsites.net",
    "http://csimistral.azurewebsites.net",


]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_response(inst, max_new_tokens=100):
    
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
    ### Input: {inst} [/INST]
    """
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9,temperature=0.9)

    res = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0] 
    
    split_res = res.split("[/INST]")
    res = split_res[-1]

    return res

@app.post("/generate")
async def generate_json(instruction: Instructions):  
    res = generate_response(instruction)
    return {"success" : True, "json_res" : res},  200


 



    
    
