import numpy as np
from PIL import Image
from centerline_lab.diagnostics import RunRecorder,files_for_step


def test_step_artifacts_and_export(tmp_path):
    image=Image.new('RGB',(64,64),'white')
    points=np.array([[9,25],[30,30],[53,33]],float)
    r=RunRecorder(tmp_path,image,points)
    proba=np.zeros((64,64),dtype=np.float32)
    proba[26:36,9:54]=1
    r.record(1,.9,image,proba,(0,0,64,64),{'loss_total':.6,'loss_skeleton_to_stroke':.2,
             'loss_stroke_to_skeleton':.3},guided=False)
    archive=r.finish(image,{'model':'mock'})
    assert archive.exists()
    images,metrics=files_for_step(str(tmp_path),1)
    assert len(images)==8 and metrics['step']==1
    assert (tmp_path/'loss.png').exists()
    assert (tmp_path/'alignment.png').exists()
    assert (tmp_path/'steps/step_001_raw_clipseg.png').exists()
    assert (tmp_path/'steps/step_001_change_prior.png').exists()
    assert (tmp_path/'metrics.csv').exists()


def test_hard_skeleton_junctions_and_empty_masks():
    from centerline_lab.diagnostics import junction_stats, hard_skeleton_metrics
    line=np.zeros((64,64),bool);line[32,10:55]=True
    tee=line.copy();tee[32:53,32]=True
    assert junction_stats(line)['hard_branchpoints']==0
    assert junction_stats(tee)['hard_branchpoints']>=1
    assert junction_stats(tee)['hard_endpoints']>=3
    empty=hard_skeleton_metrics(np.zeros((64,64)),np.array([[9,32],[55,32]]), (0,0,64,64))
    assert empty['hard_skeleton_to_stroke_px'] is None
    assert empty['hard_branchpoints']==0
