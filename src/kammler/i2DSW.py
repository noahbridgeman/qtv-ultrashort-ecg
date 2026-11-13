"""Placeholder for the i2DSW implementation.

The iterative two-dimensional signal warping (i2DSW) algorithm (Schmidt et al.
2014, 2018) used in this project is not included in this repository. This stub
only keeps the interface that ``src/pipeline/qtv_i2dsw.py`` calls, so the rest
of the pipeline (preprocessing, template building, NeuroKit2 baseline, metrics,
synthetic data) stays importable and runnable.

To run the i2DSW path, replace this file with an implementation that provides
``Optimizer.execute_i2DSW`` with the signature below and set ``AVAILABLE`` to
``True``.
"""

AVAILABLE = False

_MESSAGE = ("The i2DSW implementation is not included in this repository; "
            "see src/kammler/i2DSW.py.")


class Optimizer:
    """Interface of the i2DSW optimizer used by ``qtv_i2dsw.py``."""

    @staticmethod
    def execute_i2DSW(template, waveform, q_on_index, t_end_index, iter_step_max,
                      x_warping_bound=0.3, y_warping_bound=0.3, visualize_steps=False,
                      breaking_conditions_flavour="MATLAB_code"):
        """Warp ``template`` onto ``waveform``.

        Returns:
            (warped_template, q_on_index, t_end_index, original_distance,
            final_distance, iter_step)
        """
        raise NotImplementedError(_MESSAGE)
