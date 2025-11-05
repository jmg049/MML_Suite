  Your results show:
  - accuracy_AI (both modalities): 0.922 ✅ (expected to be high)
  - accuracy_A (audio only): 0.044 ❌ (should be reasonable since audio is available)
  - accuracy_I (image only): 0.922 ❌ (should be very low since image is supposed to be missing!)

  This pattern strongly suggests the modality conditions are inverted.

  Most Likely Root Causes

  1. C-MAM Configuration Inversion (HIGH PROBABILITY)

  The C-MAM might be configured backwards:
  - Expected: Input=Audio → Generate=Image (for audio-only client)
  - Actual: Input=Image → Generate=Audio

  This would explain why "image-only" metrics perform well when the client supposedly has "audio-only" data.

  2. Missing Modality Simulation Logic Inversion (HIGH PROBABILITY)

  The data masking during C-MAM testing might be inverted:
  - Expected: Keep audio, mask image → Test audio-only performance
  - Actual: Keep image, mask audio → Test image-only performance (but label it as audio-only)

  3. Metric Labeling Logic Error (MEDIUM PROBABILITY)

  The modality suffixes (_A, _I, _AI) might be assigned incorrectly:
  - The code might be swapping which metrics get which suffix
  - The miss_type tracking could be inverted

  Where to Investigate

  1. Client Assignment Logic

  Check how clients are assigned their missing modality:
  - Is available_modalities correctly set?
  - Does "A" really mean "audio-only" or is it backwards?

  2. C-MAM Input/Target Configuration

  Look at how C-MAMs are instantiated:
  - Are input_modalities and target_modality correctly assigned?
  - For an "audio-only" client, the C-MAM should have input_modalities=['audio'] and target_modality='image'

  3. Data Masking in C-MAM Testing

  Examine the _test_single_cmam method and how it applies missing modality patterns:
  - Is the masking logic correctly zeroing the missing modality?
  - Are the modality tensors being masked correctly?

  4. Metric Calculation Logic

  Check how the modality-specific metrics are computed:
  - How does the code determine which samples belong to _A vs _I vs _AI?
  - Is the miss_type array correctly reflecting the actual missing pattern?

  My Hypothesis

  Based on the systematic nature of this reversal, I suspect the issue is in C-MAM configuration or data
  masking logic. The fact that:
  - Main model metrics are trustworthy
  - Only C-MAM metrics show this pattern
  - The reversal is consistent across all metrics

  Suggests the problem is specific to how C-MAMs handle modality conditions, not in the overall experimental
   setup.

  The most likely scenario: An "audio-only" client is getting a C-MAM that reconstructs audio from image
  (backwards), or the test data is being masked incorrectly so that image data is kept when audio should be
  kept.