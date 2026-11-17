"""Single-screen research app: scribble -> Klein+soft skeleton feedback -> inspect."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
from uuid import uuid4

import gradio as gr

from .sampler import RunConfig, run_generation
from .diagnostics import files_for_step


def make_run_dir():
    name = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-')+uuid4().hex[:8]
    return Path('outputs')/name


def inspect_step(state: dict, step: int):
    if not state or not state.get('dir'):
        return (*([None]*8), 'Run generation first.')
    images,metrics=files_for_step(state['dir'],int(step))
    compact={k:metrics.get(k) for k in ('step','sigma','guided','feedback_applied','loss_total',
             'loss_skeleton_to_stroke','loss_stroke_to_skeleton','loss_branch',
             'loss_coverage','clipseg_confidence_proxy','clipseg_blend_weight',
             'mean_source_difference','hard_skeleton_to_stroke_px',
             'hard_stroke_to_skeleton_px','hard_branchpoints','hard_connected_components',
             'post_loss_total','post_loss_delta','post_hard_stroke_to_skeleton_px',
             'gradient_rms','update_rms','step_size')}
    return *images,json.dumps(compact,indent=2)


def run(editor, label, instruction, model_id, clipseg_id, device,
        steps, cfg, strength, seed, edge, padding, feedback, cadence,
        sigma_min,sigma_max,max_update,context,skel_iters,geom_size,
        branch_weight,coverage_weight,early_prior,post_check,
        progress=gr.Progress(track_tqdm=False)):
    config=RunConfig(object_label=label,instruction=instruction,model_id=model_id,
                     clipseg_id=clipseg_id,device=device,steps=int(steps),cfg=float(cfg),
                     inpaint_strength=float(strength),seed=int(seed),max_edge=int(edge),
                     region_padding=int(padding),feedback_strength=float(feedback),
                     feedback_every=int(cadence),sigma_min=float(sigma_min),
                     sigma_max=float(sigma_max),max_update=float(max_update),
                     crop_context=float(context),skeleton_iterations=int(skel_iters),
                     loss_resolution=int(geom_size),branch_weight=float(branch_weight),
                     coverage_weight=float(coverage_weight),early_change_prior=bool(early_prior),
                     check_post_feedback=bool(post_check))
    folder=make_run_dir()
    try:
        progress(0.02,desc='Loading frozen Klein and CLIPSeg')
        def on_step(i,total):
            progress(min(.98,.05+.9*i/total),desc=f'Denoising + inspecting {i}/{total}')
        result,rec,meta,archive=run_generation(editor,config,folder,on_step=on_step)
        progress(1.,desc='Saved final image and diagnostics')
        status=(f"**Completed:** {meta['actual_steps']} denoising steps, "
                f"{meta['gradient_steps']} nonzero feedback corrections. "
                "Saved pre-update clean estimates and segmentation/skeleton maps for each step. "
                "The *final image* is Klein's raw output, not a composited mask.")
        import numpy as np
        from PIL import Image
        from .diagnostics import overlay_scribble
        source=Image.open(folder/'source.png').convert('RGB')
        points=np.load(folder/'scribble_points.npy')
        roi=Image.open(folder/'inpaint_mask.png').convert('L')
        overlay=overlay_scribble(source,points)
        metrics=rec.history
        table_keys=['step','sigma','guided','loss_total','loss_skeleton_to_stroke',
                    'loss_stroke_to_skeleton','loss_branch','clipseg_blend_weight',
                    'hard_branchpoints','hard_stroke_to_skeleton_px','post_hard_stroke_to_skeleton_px',
                    'gradient_rms','update_rms']
        rows=[[round(v,5) if isinstance(v,float) else v for v in (r.get(k) for k in table_keys)]
              for r in metrics]
        first=inspect_step({'dir':str(folder)},1)
        return ({'dir':str(folder)},result,overlay,roi,rec.gallery,
                str(folder/'loss.png'),str(folder/'alignment.png'),rows,status,str(archive),
                gr.update(value=1,maximum=max(2,meta['actual_steps'])),*first)
    except Exception as exc:
        raise gr.Error(f'{type(exc).__name__}: {exc}') from exc


def make_ui():
    with gr.Blocks(title='Centerline Flow Lab · Klein') as demo:
        gr.Markdown('## Centerline Flow Lab\n'
                    '**One pass:** original image + object prompt + one centerline scribble → '
                    'FLUX.2 Klein inpainting with differentiable soft-skeleton feedback. '
                    'Every denoising step is saved for inspection. No SAM 3, shape adaptation, or multi-stage generation.')
        state=gr.State({})
        with gr.Row():
            with gr.Column(scale=5):
                editor=gr.ImageEditor(label='Upload image and draw ONE open centerline',type='pil',
                    image_mode='RGBA',sources=['upload','clipboard'],
                    brush=gr.Brush(default_size=6,colors=['#F44336'],color_mode='fixed'),
                    layers=False,height=420)
                with gr.Row():
                    label=gr.Textbox(label='Object label (CLIPSeg and Klein)',value='a snake')
                    instruction=gr.Textbox(label='Insertion instruction',value='Add one realistic snake on the sofa with natural anatomy and perspective.',lines=2)
                with gr.Accordion('Klein inference',open=True):
                    model=gr.Textbox(label='Model ID or local Diffusers directory',
                                      value='black-forest-labs/FLUX.2-klein-base-4B')
                    with gr.Row():
                        edge=gr.Dropdown([512,768,1024],value=512,label='Max image edge')
                        padding=gr.Slider(16,320,value=80,step=8,label='Editable-region padding (px)')
                        device=gr.Dropdown(['auto','mps','cuda','cpu'],value='auto',label='Device')
                    with gr.Row():
                        steps=gr.Slider(4,60,value=30,step=1,label='Denoising steps')
                        cfg=gr.Slider(1,8,value=4,step=.5,label='CFG (Base model)')
                    with gr.Row():
                        strength=gr.Slider(.5,1,value=1,step=.05,label='Inpaint strength')
                        seed=gr.Number(value=42,precision=0,label='Seed')
                with gr.Accordion('Skeleton feedback',open=True):
                    with gr.Row():
                        feedback=gr.Slider(0,4,value=.75,step=.05,label='Feedback strength (0 = baseline)')
                        cadence=gr.Slider(1,8,value=2,step=1,label='Correct every N steps')
                    with gr.Row():
                        sigma_min=gr.Slider(0,.9,value=.10,step=.05,label='Minimum sigma')
                        sigma_max=gr.Slider(.1,1,value=.98,step=.01,label='Maximum sigma (earlier feedback)')
                        max_update=gr.Slider(.01,.40,value=.10,step=.01,label='Maximum latent step')
                    with gr.Row():
                        skel_iters=gr.Slider(6,40,value=18,step=1,label='Soft skeleton iterations')
                        geom_size=gr.Dropdown([64,96,112,128,160],value=112,label='Geometry loss grid')
                        context=gr.Slider(.15,1,value=.35,step=.05,label='CLIPSeg crop context')
                    with gr.Row():
                        branch_weight=gr.Slider(0,2,value=.25,step=.05,label='Branch/junction penalty')
                        coverage_weight=gr.Slider(0,1,value=.04,step=.01,label='Foreground coverage weight')
                    with gr.Row():
                        early_prior=gr.Checkbox(value=True,label='Early fallback: changed-region proxy (not segmentation)')
                        post_check=gr.Checkbox(value=True,label='Measure immediate post-correction geometry (slower)')
                    clipseg=gr.Textbox(label='Frozen segmentation model',value='CIDAS/clipseg-rd64-refined')
                run_btn=gr.Button('Generate + record every denoising step',variant='primary')
            with gr.Column(scale=6):
                result=gr.Image(label='Final raw Klein output',type='pil')
                with gr.Row():
                    stroke_display=gr.Image(label='Original + centerline',type='pil')
                    edit_region=gr.Image(label='Inpainting region (white editable)',type='pil')
                status=gr.Markdown('Not run')
                loss_img=gr.Image(label='Feedback surrogate loss (not actual geometry)',type='filepath')
                align_img=gr.Image(label='Independent hard-skeleton alignment (px; lower is better)',type='filepath')
                gallery=gr.Gallery(label='Per-step overlay: red=user scribble, green=predicted hard skeleton',
                                   columns=4,object_fit='contain',height=280)
                step_slider=gr.Slider(1,2,value=1,step=1,label='Inspect denoising step')
                with gr.Row():
                    predicted=gr.Image(label='Predicted clean image BEFORE correction',type='pil')
                    probability=gr.Image(label='Blended guidance probability (used for skeleton)',type='pil')
                with gr.Row():
                    soft_skeleton=gr.Image(label='Soft skeleton USED for gradient',type='pil')
                    skeleton=gr.Image(label='Hard skeleton (diagnostic only)',type='pil')
                with gr.Row():
                    update_map=gr.Image(label='Spatial latent-update magnitude',type='pil')
                    composite=gr.Image(label='Scribble vs predicted skeleton',type='pil')
                with gr.Row():
                    raw_clipseg=gr.Image(label='Raw CLIPSeg probability',type='pil')
                    change_preview=gr.Image(label='Early source-change proxy (not semantic)',type='pil')
                metrics_preview=gr.Code(label='Selected-step metrics',language='json')
                table=gr.Dataframe(headers=['step','sigma','guided','loss','skeleton→stroke','stroke→skeleton',
                                     'junction loss','CLIPSeg weight','hard junctions',
                                     'hard stroke→skeleton px','post stroke→skeleton px',
                                     'gradient RMS','update RMS'],
                                   label='All-step measurements',interactive=False)
                archive=gr.File(label='Complete research run ZIP',interactive=False)
        controls=[editor,label,instruction,model,clipseg,device,steps,cfg,strength,seed,
                  edge,padding,feedback,cadence,sigma_min,sigma_max,max_update,context,skel_iters,geom_size,
                  branch_weight,coverage_weight,early_prior,post_check]
        outputs=[state,result,stroke_display,edit_region,gallery,loss_img,align_img,table,status,archive,
                 step_slider,predicted,probability,soft_skeleton,skeleton,update_map,composite,raw_clipseg,change_preview,metrics_preview]
        run_btn.click(run,inputs=controls,outputs=outputs)
        step_slider.change(inspect_step,inputs=[state,step_slider],
                           outputs=[predicted,probability,soft_skeleton,skeleton,update_map,composite,raw_clipseg,change_preview,metrics_preview])
        gr.Markdown('**Interpretation:** each intermediate image is Klein’s *predicted clean image*, '
                    'not the actual completed image at that step. CLIPSeg may produce unreliable masks '
                    'early in sampling. The image-change proxy can react to shadows or unrelated edits, '
                    'so its blended guidance is experimental. Hard skeletons and hard distances are independent '
                    'diagnostics; guidance uses the differentiable soft skeleton. '
                    'Post-update checks reuse the previous model velocity. Set feedback to 0 for a same-seed baseline. '
                    'Apple MPS autograd through the VAE is experimental.')
    return demo


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=7860)
    args=parser.parse_args()
    make_ui().queue(default_concurrency_limit=1).launch(server_name=args.host,
                                                        server_port=args.port,show_error=True)

if __name__=='__main__':main()
