def run_smooth(*args, **kwargs):
    from pedi_oku_landslide.pipeline.steps.step_smooth import run_smooth as _impl

    return _impl(*args, **kwargs)


def run_sad(*args, **kwargs):
    from pedi_oku_landslide.pipeline.steps.step_sad import run_sad as _impl

    return _impl(*args, **kwargs)


def run_detect(*args, **kwargs):
    from pedi_oku_landslide.pipeline.steps.step_detect import run_detect as _impl

    return _impl(*args, **kwargs)


def render_vectors(*args, **kwargs):
    from pedi_oku_landslide.pipeline.steps.step_detect import render_vectors as _impl

    return _impl(*args, **kwargs)


def run_mask_from_dxf(*args, **kwargs):
    from pedi_oku_landslide.pipeline.steps.step_mask_dxf import run_mask_from_dxf as _impl

    return _impl(*args, **kwargs)


__all__ = ["render_vectors", "run_detect", "run_mask_from_dxf", "run_sad", "run_smooth"]
