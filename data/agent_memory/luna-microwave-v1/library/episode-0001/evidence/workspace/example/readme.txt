Worked example supplied by the human researcher: refine a hinge from motion.

An RGB-selected hinge edge gives an initial estimate. After a planned motion,
measured door landmarks can disagree with that estimate. Hand motion alone does
not establish door motion because contact can slip or push the wrong surface.

In this earlier development example, we tracked material image corners before
and after a short physical motion, and lifted them through metric depth and
camera calibration. The supplied matches.npz contains reference_world and
current_world arrays from those sensor correspondences. The two PNGs show the
camera before and after the corresponding closure motion.

For an upright mechanism, fit p_after_xy = R p_before_xy + t to consistent moving
points. A fixed pivot h satisfies (I-R)h=t. Outliers, stationary background and
insufficient rotation make this unreliable. Check residuals, spatial support,
rotation magnitude and stability under resampling before trusting h. A reliable
rigid transform can still update the moving cloud when its pivot is uncertain.

The old experiment used observations to revise the model, then planned and
executed subsequent motions. This is an example of a diagnostic process, not a
provided trajectory for the new case. It supplies an upright-axis assumption
and the rigid-motion fitting idea. Your task is to decide whether/when that idea
is useful, choose your own regions and motions, and record what you observe.

There are no simulator joint measurements or true hinge coordinates in these
example files. The sensor-fit.json is a numerical estimate from the correspondences.
