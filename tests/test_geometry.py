import numpy as np
import torch
from centerline_lab.geometry import soft_skeletonize, CenterlineLoss, LossSettings


def test_soft_skeleton_shape_and_gradient():
    x=torch.zeros((1,1,48,48),requires_grad=True)
    with torch.no_grad():
        x[:,:,15:33,21:27]=.8
    y=soft_skeletonize(x,iterations=15)
    assert y.shape==x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_bidirectional_loss_prefers_aligned_skeleton():
    points=np.array([[.2,.5],[.3,.5],[.4,.5],[.5,.5],[.6,.5],[.7,.5],[.8,.5]],np.float32)
    loss_fn=CenterlineLoss(points,'cpu',LossSettings(resolution=64,skeleton_iterations=10))
    yy,xx=torch.meshgrid(torch.linspace(0,1,64),torch.linspace(0,1,64),indexing='ij')
    # Ribbon-shaped foreground near / far from user's stroke.
    aligned=(torch.exp(-((yy-.5)**2)/(.07**2))*torch.sigmoid((xx-.2)*45)*torch.sigmoid((.8-xx)*45))
    shifted=(torch.exp(-((yy-.18)**2)/(.07**2))*torch.sigmoid((xx-.2)*45)*torch.sigmoid((.8-xx)*45))
    def logit(x):return torch.logit(x.clamp(.001,.999))[None,None]
    a=loss_fn(logit(aligned))[0]
    b=loss_fn(logit(shifted))[0]
    assert a<b,(a,b)


def test_centerline_loss_backpropagates():
    target=np.array([[.2,.5],[.5,.5],[.8,.5]],np.float32)
    f=CenterlineLoss(target,'cpu',LossSettings(resolution=64,skeleton_iterations=6))
    x=torch.randn((1,1,64,64),requires_grad=True)
    loss,metrics,*_=f(x)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert any(k.startswith('loss_') for k in metrics)


def test_t_junction_has_larger_differentiable_branch_penalty():
    from centerline_lab.geometry import branchpoint_surrogate
    line = torch.zeros((1,1,64,64),requires_grad=True)
    with torch.no_grad():
        line[:, :, 32, 12:52] = 1
    tee = line.detach().clone().requires_grad_(True)
    with torch.no_grad():
        tee[:, :, 32:55, 32] = 1
    a = branchpoint_surrogate(line)
    b = branchpoint_surrogate(tee)
    assert b.item() > a.item() * 3, (a,b)
    b.backward()
    assert tee.grad is not None and torch.isfinite(tee.grad).all()


def test_source_change_fallback_has_grad_and_is_not_clips_seg():
    from centerline_lab.geometry import changed_region_prior
    original = torch.full((1,3,64,64), .5)
    generated = original.clone()
    generated[:,:,23:41,12:52] = .05
    generated.requires_grad_(True)
    allowed = torch.ones((1,1,64,64))
    prior,change = changed_region_prior(generated,original,(0,0,64,64),allowed,64)
    assert change > .01
    assert prior[:,:,27:37,20:44].mean() > prior[:,:,:12,:].mean()
    prior.sum().backward()
    assert generated.grad is not None and generated.grad.abs().sum() > 0


def test_low_clipseg_confidence_uses_change_fallback():
    from centerline_lab.geometry import segmentation_reliability
    target = np.array([[.2,.5],[.5,.5],[.8,.5]],np.float32)
    objective = CenterlineLoss(target,'cpu',LossSettings(resolution=64,skeleton_iterations=6))
    bad_logits=torch.full((1,1,64,64), -12.)
    prior=torch.zeros((1,1,64,64))
    prior[:,:,28:36,10:53] = .9
    no_prior_loss,no_prior_stats,_,_=objective(bad_logits)
    with_prior_loss,stats,blended,_=objective(bad_logits,change_prior=prior)
    assert stats['clipseg_blend_weight'] < .1
    assert blended.max() > .7
    assert stats['raw_clipseg_peak'] < .01
    assert torch.isfinite(with_prior_loss)
    assert stats['loss_skeleton_to_stroke'] < no_prior_stats['loss_skeleton_to_stroke']


def test_strong_clipseg_disables_early_fallback():
    target = np.array([[.2,.5],[.5,.5],[.8,.5]],np.float32)
    objective = CenterlineLoss(target,'cpu',LossSettings(resolution=64,skeleton_iterations=6))
    logits=torch.full((1,1,64,64), -6.)
    logits[:,:,27:37,10:53] = 8.
    wrong_prior=torch.ones((1,1,64,64))*.9
    _,stats,blended,_=objective(logits,change_prior=wrong_prior)
    assert stats['clipseg_blend_weight'] > .9
    assert blended[:,:,0:10,:].mean() < .1
