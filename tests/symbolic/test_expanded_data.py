import numpy as np
from structure_screen.expanded_data import estimated_r2
from structure_screen.bounded_model import BoundedStructureRegressor
from structure_screen.train_expanded import clip_r2_score
import torch

def test_estimated_r2_handles_constant_and_collinear_columns():
    y=np.arange(20,dtype=float); p=y*2; score=estimated_r2([np.ones(20),p,p],y)
    assert score is not None and 0 <= score <= 1 and score > .999

def test_estimated_r2_rejects_nan():
    assert estimated_r2([np.full(10,np.nan)],np.arange(10.0)) is None

def test_network_output_never_exceeds_one():
    model=BoundedStructureRegressor(10,8,12); result=model(torch.randint(1,10,(16,7)))
    assert torch.isfinite(result).all() and torch.all(result <= 1.0)

def test_negative_r2_is_retained_and_extreme_values_are_clipped():
    config={"r2_clip_min":-10.0,"r2_clip_max":1.0}
    assert clip_r2_score(-2.5,config) == -2.5
    assert clip_r2_score(-999.0,config) == -10.0
    assert clip_r2_score(1.2,config) == 1.0
