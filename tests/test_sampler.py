import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from centerline_lab.sampler import RunConfig,SamplerInspector,bbox_crop,normalized_crop_points
from centerline_lab.diagnostics import RunRecorder

class FakeScheduler:
    def __init__(self):
        self.sigmas=torch.tensor([.70,.45])
        self.step_index=0
    def step(self,velocity,timestep,sample,*args,**kwargs):
        self.step_index+=1
        return (sample-.25*velocity,)
class FakePipe:
    def __init__(self):
        self.scheduler=FakeScheduler()
        self.vae_scale_factor=8
class FakeSegmenter:
    def logits(self,image,rect):
        x0,y0,x1,y1=rect
        cut=image[:,:,y0:y1,x0:x1]
        return F.interpolate(cut.mean(1,keepdim=True)*8-4,(64,64),mode='bilinear',align_corners=False)


def install_fake_decoder(inspector):
    def decode(tokens):
        b,n,c=tokens.shape
        z=tokens.mean(-1).reshape(b,1,4,4)
        x=torch.sigmoid(F.interpolate(z,(64,64),mode='bilinear',align_corners=False))
        return x.repeat(1,3,1,1)
    inspector.decode_predicted_x0=decode


def construct(tmp_path,guidance):
    src=Image.new('RGB',(64,64),'white')
    stroke=np.array([[10.,30.],[30.,30.],[52.,30.]],float)
    edit=Image.new('L',(64,64),255)
    cfg=RunConfig(steps=1,feedback_strength=guidance,feedback_every=1,
                  sigma_min=.1,sigma_max=.9,loss_resolution=64,skeleton_iterations=6)
    ins=SamplerInspector(FakePipe(),FakeSegmenter(),RunRecorder(tmp_path,src,stroke),
                          stroke,edit,'cpu',cfg)
    install_fake_decoder(ins)
    return ins


def test_scheduler_feedback_moves_latents_and_records(tmp_path):
    ins=construct(tmp_path,.75)
    initial=torch.randn((1,16,4))*0.5
    velocity=torch.ones_like(initial)
    with ins:
        guided=ins.pipe.scheduler.step(velocity,.70,initial,return_dict=False)[0]
    baseline=initial-.25*velocity
    assert not torch.allclose(guided,baseline)
    assert ins.guidance_steps==1
    assert ins.recorder.history[0]['feedback_applied']
    assert (tmp_path/'steps/step_001_pred.png').is_file()
    assert (tmp_path/'steps/step_001_skeleton.png').is_file()
    assert (tmp_path/'steps/step_001_soft_skeleton.png').is_file()
    assert (tmp_path/'steps/step_001_update.png').is_file()


def test_guidance_zero_same_scheduler_and_still_visualizes(tmp_path):
    ins=construct(tmp_path,0)
    x=torch.randn((1,16,4))
    v=torch.ones_like(x)
    with ins:
        out=ins.pipe.scheduler.step(v,.70,x,return_dict=False)[0]
    assert torch.allclose(out,x-.25*v)
    assert ins.guidance_steps==0
    assert len(ins.recorder.history)==1
    assert not ins.recorder.history[0]['feedback_applied']


def test_crop_point_normalization():
    pts=np.array([[10,10],[45,42]],float)
    rect=bbox_crop(pts,(64,64),.35)
    xy=normalized_crop_points(pts,rect)
    assert xy.shape==(2,2) and np.all((xy>=0)&(xy<=1))


def test_scheduler_feedback_inside_no_grad(tmp_path):
    """A scheduler callback must reopen grad locally inside Diffusers."""
    ins = construct(tmp_path, .75)
    x = torch.randn((1,16,4))
    v = torch.ones_like(x)
    with torch.no_grad():
        with ins:
            guided = ins.pipe.scheduler.step(v, .70, x, return_dict=False)[0]
    assert not torch.allclose(guided, x - .25 * v)
    assert ins.guidance_steps == 1


def test_scheduler_feedback_inside_inference_mode(tmp_path):
    """enable_grad alone is insufficient in inference_mode; leaf must be cloned."""
    ins = construct(tmp_path, .75)
    with torch.inference_mode():
        x = torch.randn((1,16,4))
        v = torch.ones_like(x)
        with ins:
            guided = ins.pipe.scheduler.step(v, .70, x, return_dict=False)[0]
    assert not torch.allclose(guided, x - .25 * v)
    assert ins.guidance_steps == 1


def test_disconnected_segmenter_raises_actionable_error(tmp_path):
    class DisconnectedSegmenter:
        def logits(self, image, rect):
            return image.mean(dim=1,keepdim=True).detach()[:, :, 0:32, 0:32]
    ins = construct(tmp_path, .75)
    ins.segmenter = DisconnectedSegmenter()
    with ins:
        import pytest
        with pytest.raises(RuntimeError, match='Gradient path disconnected'):
            ins.pipe.scheduler.step(torch.ones((1,16,4)), .70,
                                    torch.randn((1,16,4)), return_dict=False)


def test_post_feedback_evaluated_and_uses_applied_delta(tmp_path):
    ins=construct(tmp_path,.75)
    x=torch.randn((1,16,4))
    v=torch.ones_like(x)
    with ins:
        ins.pipe.scheduler.step(v,.70,x,return_dict=False)
    metrics=ins.recorder.history[0]
    assert metrics['feedback_applied']
    assert metrics['post_loss_total'] is not None
    assert np.isfinite(metrics['post_loss_total'])
    assert metrics['post_loss_delta'] is not None
    assert metrics['clipseg_blend_weight']>=0
    assert (tmp_path/'steps/step_001_raw_clipseg.png').exists()
    assert (tmp_path/'steps/step_001_change_prior.png').exists()


def test_early_feedback_disabled_produces_only_clips_seg_guidance(tmp_path):
    ins=construct(tmp_path,.75)
    ins.config.early_change_prior=False
    ins.config.check_post_feedback=False
    with ins:
        ins.pipe.scheduler.step(torch.ones((1,16,4)),.70,torch.randn((1,16,4)),return_dict=False)
    metrics=ins.recorder.history[0]
    assert metrics['clipseg_blend_weight']==1.0
    assert metrics['used_change_prior'] is False
    assert metrics['post_loss_total'] is None
