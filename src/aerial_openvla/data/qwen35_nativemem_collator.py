"""Native Qwen image-chat layout backed by cached pre-merger tensors."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any
import torch
from PIL import Image

@dataclass(slots=True)
class Qwen35NativeMEMCollator:
    processor: Any
    prompt_template: str
    max_text_tokens: int=256
    _dummy_image: Any=field(init=False,repr=False)
    def __post_init__(self):
        native_pixels=224*224
        self.processor.image_processor.size['shortest_edge']=native_pixels
        self.processor.image_processor.size['longest_edge']=native_pixels
        self._dummy_image=Image.new('RGB',(224,224))
    def __call__(self, examples):
        conversations=[]
        for e in examples:
            prompt=self.prompt_template.format(instruction=e['instruction'].strip())
            conversations.append([{'role':'user','content':[{'type':'image','image':self._dummy_image},{'type':'text','text':prompt}]}])
        encoded=self.processor.apply_chat_template(conversations,tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt',processor_kwargs={'padding':True,'truncation':True,'max_length':self.max_text_tokens})
        required=('input_ids','attention_mask','image_grid_thw','mm_token_type_ids')
        missing=[key for key in required if key not in encoded]
        if missing: raise RuntimeError(f'NativeMEM processor omitted native alignment fields: {missing}')
        if tuple(encoded['image_grid_thw'].shape)!=(len(examples),3): raise RuntimeError('NativeMEM requires one current image grid per sample')
        return {**{key:encoded[key] for key in required},'nativemem_premerger_features':torch.stack([e['nativemem_premerger_features'] for e in examples]),'nativemem_history_valid_mask':torch.stack([e['nativemem_history_valid_mask'] for e in examples]),'action_id':torch.tensor([e['action_id'] for e in examples],dtype=torch.long),'episode_id':[e['episode_id'] for e in examples],'trajectory_id':[e['trajectory_id'] for e in examples],'step_id':[e['step_id'] for e in examples],'current_visual_cache_row':[e['current_visual_cache_row'] for e in examples]}
