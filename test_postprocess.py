from nvsr_unet import postprocessing2, postprocessing
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
import torch
import numpy as np
@settings(max_examples=8, deadline=2000)
@given(arrays(np.float32, (1<<13,), elements=st.floats(min_value=-1, max_value=1, allow_nan=False, width=32)), arrays(np.float32, (1<<13,), elements=st.floats(min_value=-1, max_value=1, allow_nan=False, width=32)))
def test_postprocessing(x, out):
    np.allclose(postprocessing(x, out), postprocessing2(torch.from_numpy(x), torch.from_numpy(out)).numpy())