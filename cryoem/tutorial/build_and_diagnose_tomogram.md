# Cryo-ET: build and diagnose a cryo-ET tomogram

Task: You're going to do three things: stack → align → reconstruct.

**Dataset (raw tilt movies + SerialEM metadata)**

Use the 5-tilt-series subset of EMPIAR-10164, which the teamtomo walkthrough provides a simple download script for (it pulls .mdoc metadata + the per-tilt multi-frame .mrc files).

Link: <https://teamtomo.org/teamtomo-site-archive/walkthroughs/EMPIAR-10164/preparation.html>

**Instructions**

Please answer all questions and upload images from the software to verify your work.

**Setup**

1. Download ONE tilt series: TS_01

2. Install **IMOD/Etomo** to use **3dmod** (viewer) and **Etomo** (alignment + reconstruction pipeline)

**Steps:**

**Step A: Make a tilt stack**

Combine the individual tilt images into one stack that Etomo can manage.

In practice, you'll create an IMOD project and import the tilt series. Workflows differ depending on whether you preprocess movies elsewhere (Warp) or not; teamtomo uses Warp for motion correction and then exports an IMOD stack.

If you have Warp available: follow the teamtomo "Stack generation" section (Warp's "import tilt series from IMOD" and "Create stacks for IMOD").

If you're IMOD-only (no Warp): you can still proceed, but motion-correction quality may be worse.

**Step B: Align the tilt series (fiducials)**

This dataset uses gold beads (fiducials). Your job is to track them and minimize residual error.

In Etomo:

- Choose fiducial-based alignment
- Track beads through the tilt range
- Iterate until residuals stabilize

**Step C: Reconstruct TWO tomograms at the missing wedge**

- WBP (weighted back projection)
- SIRT (iterative reconstruction)

**Questions**

**Q1: Alignment quality**

In Etomo, record final fiducial alignment residual

**Answer:**

- What is the final residual error (in pixels)?
- Which tilts contribute the worst residuals (high tilt? low tilt?) and why?

**Q2: Missing wedge**

In 3dmod: rotate your tomogram and look for direction-dependent blur/elongation (post an image of your work)

Reconstruct a second time after excluding all tilts beyond ±40° (Etomo lets you exclude views) (post an image of your work)

**Answer:**

- Which features distort most as you worsen the angular coverage?
- Describe the artifact in _Z vs XY_.

**Q3: Compare the WBP and SIRT tomograms**

Pick the same ROI (membranes/beads/background)

**Answer:**

- Which method shows less streaking?
- Which method looks "sharper" vs "smoother"?
- Which one would you trust for template matching vs segmentation?

**Q4: In Etomo/IMOD, note the pixel size of your reconstructed tomogram.**

**Answer:**

- What is the Nyquist limit (in Å) for your tomogram?
- Based on what you _see_, are you anywhere near that limit?
