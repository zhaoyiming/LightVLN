import os,json,hashlib,runpy
from pathlib import Path
import torch
from peft import PeftModel
import aerial_openvla.models.airminivla_classifier as classifiers

original=classifiers.build_airminivla_classifier
def build(*args,**kwargs):
    processor,m=original(*args,**kwargs)
    m.eval();torch.backends.cuda.matmul.allow_tf32=False
    ck=Path(kwargs['checkpoint_dir'])
    state=torch.load(ck/'training_state.pt',map_location='cpu',weights_only=False)
    assert state['global_step'] in (13563,18084)
    c=m.config
    assert c.visual_tokens_per_frame==32 and c.native1t_history_length==16
    assert c.local_history_variant=='instruction_selection_residual'
    assert c.openfly_action_policy is None
    lm=m.backbone.language_model
    assert isinstance(lm,PeftModel) and lm.active_adapters==['default']
    device=next(lm.parameters()).device
    ids=torch.tensor([[1,100,200,300]],device=device)
    with torch.inference_mode():
        def forward():return lm(input_ids=ids,output_hidden_states=True,use_cache=False).hidden_states[-1].float()
        active=forward()
        with lm.disable_adapter(): base=forward()
        effect=(active-base).abs().max().item();assert effect>0
        target=next(x for x in lm.modules() if hasattr(x,'lora_A') and len(x.lora_A))
        weight=target.get_base_layer().weight
        before=weight.detach().clone()
        merged=lm.merge_and_unload(safe_merge=True)
        changed=(weight-before).abs().max().item();assert changed>0
        m.backbone.language_model=merged
        after=merged(input_ids=ids,output_hidden_states=True,use_cache=False).hidden_states[-1].float()
        torch.testing.assert_close(after,active,atol=.003,rtol=.003)
    report=dict(checkpoint=str(ck),step=state['global_step'],mode='merged',adapter_effect=effect,merged_weight_change=changed,merge_max_abs=(after-active).abs().max().item(),current_tokens=32,history=16,variant=c.local_history_variant,adapter_sha256=hashlib.sha256((ck/'lora/adapter_model.safetensors').read_bytes()).hexdigest())
    dest=Path(os.environ['AIRVLN_AUDIT_DIR']);dest.mkdir(parents=True,exist_ok=True)
    (dest/('lora_load_report_worker'+os.environ.get('AIRVLN_WORKER_ID','0')+'.json')).write_text(json.dumps(report,indent=2))
    def pool_hook(module,inputs,output):
        assert output.shape[1:]==(32,896),output.shape
    m.instruction_local_pooler.register_forward_hook(pool_hook)
    print('HRIS_CHECKPOINT_MERGE_PASS',json.dumps(report),flush=True)
    return processor,m
classifiers.build_airminivla_classifier=build
if __name__=='__main__':
    from aerial_openvla.evaluation.rollout import main
    main()
