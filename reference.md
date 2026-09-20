Goal
Build the smallest useful prototype that turns a video into persistent feature-track hypotheses and lets us inspect whether SuperPoint descriptors are stable enough over time to support online landmark clustering.

Pipeline
video
  ↓
stream decode frames
  ↓
SuperPoint
  ↓
~1000 keypoints + 256-D normalized descriptors/frame
  ↓
online landmark association
  ↓
persistent track IDs
  ↓
visualization
Do not extract the video to individual image files. Stream frames from the compressed video.

Data model
Each observation:

frame_id
timestamp
x, y
SuperPoint score
descriptor[256]
landmark_id
Each landmark initially stores:

id
observation_count
resultant_vector = Σ descriptor_i
mean_direction = resultant / |resultant|
concentration
list of observation IDs
Because SuperPoint descriptors are L2-normalized, treat them as points on a hypersphere.

Start with a single von Mises–Fisher-like cluster per landmark rather than a Euclidean Gaussian.

Online association
Frame 0:

Every detected feature becomes a new landmark.

For each subsequent frame:

Extract SuperPoint features.

Compare every new descriptor against existing landmark mean directions using cosine similarity.

Assign high-confidence matches to existing landmarks.

Update those landmark statistics.

Create new landmarks for unmatched observations.

Prevent multiple observations from the same frame from being assigned to the same landmark.

Initially, keep this deliberately descriptor-only. Do not add geometric constraints yet.

Baselines
Keep enough data to later compare against:

nearest-neighbor descriptor matching

mutual nearest neighbor

LightGlue

pairwise adjacent-frame matching

exhaustive offline association on a short clip

The online clustering method should be evaluated against these rather than assumed to be better.

Visualization
First useful viewer:

video frame

current SuperPoint keypoints

color by landmark ID

short trails showing previous observations of each landmark

click/select a landmark to inspect all its observations

Useful diagnostics:

landmark lifetime

number of observations

cosine similarity to cluster mean

concentration

new vs matched landmarks per frame

Optional tonight if easy:

sample descriptors and run PCA → UMAP

color UMAP points by inferred landmark ID and by frame number

The UMAP is diagnostic only, not part of association.

Success Criterion
By the end of the prototype, we should be able to play one phone video and visually answer:

Do stable landmark identities emerge?

How quickly do false merges happen?

How stable are SuperPoint descriptors over viewpoint and blur?

 there’s a real mathematical version of your “harmony vs dissonance” intuition: phase coherence.

Suppose a landmark has multiple observed signals over time—its motion, descriptor components, patch intensities, or residuals. Take Fourier transforms:

If observations describe the same underlying physical motion, their energy should often align in frequency and especially phase.

A useful quantity is cross-spectral coherence:

where is the cross-spectrum. means the two signals have a stable phase relationship at that frequency; means they’re effectively dissonant.

For your tracking problem, I’d be especially interested in applying this to motion trajectories:

landmark A: x(t), y(t)
landmark B: x(t), y(t)
landmark C: x(t), y(t)
Static-scene points under one camera motion should exhibit strongly related temporal structure, even though their amplitudes differ because of depth and image position. A pedestrian or bad track may introduce different frequency/phase behavior.

You could even define a kind of “harmonic agreement” score:

or retain the complex cross-spectrum so phase disagreement matters explicitly.

The other interesting interpretation is wiggle residuals: first subtract the smooth camera-motion component, then Fourier-analyze the residual trajectory. Good rigid tracks might have mostly low-energy residuals; bad associations, rolling shutter, people, vibration, etc. create characteristic higher-frequency energy.

So I’d think of it as:

geometry gives the melody; Fourier coherence measures whether different observations are playing the same song.

The next thing I’d try mathematically is not descriptors directly, but 2D track trajectories and reprojection residuals over time. That’s where the frequency-domain interpretation is cleanest.


