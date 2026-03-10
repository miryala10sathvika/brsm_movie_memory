# BRSM Movie Memory Experiment - Methods Section

## EXPERIMENTAL DESIGN WRITE-UP

---

## METHOD

### Participants

A total of 171 individuals participated in this study. The first 13 participants were excluded from analysis due to a data recording error, leaving a final sample of **N = 146** participants (112 males, 34 females; *M*age = 22.0 years, *SD* = 1.9, range: 19-28 years). All participants reported normal or corrected-to-normal vision (68.5% normal, 31.5% corrected). The majority were right-handed (95.2%). 

Participants were randomly assigned to one of two between-subjects conditions: the **Natural Boundary (NB) condition** (*n* = 79) or the **Abrupt Boundary (AB) condition** (*n* = 67). All participants provided informed consent and received [course credit/monetary compensation] for their participation.

---

### Stimuli

#### Video Selection and Preparation

Forty YouTube Shorts were selected as stimulus videos for this experiment. To identify event boundaries in each video, independent annotators watched the clips and marked coarse-grained event boundaries via keypress. **Consensus boundaries** were derived by identifying temporal locations where multiple annotators agreed, establishing reliable, naturally occurring event transitions in each video.

#### Condition Manipulation

Using the consensus boundaries, two versions of each video were created to manipulate event boundary continuity:

1. **Natural Cut (NB) Condition:** Videos played continuously to their natural conclusion without interruption. Event boundaries occurred exactly when viewers would naturally expect them, preserving cognitive continuity and maintaining the integrity of event structure.

2. **Abrupt Cut (AB) Condition:** Videos were abruptly terminated 1-5 seconds *before* a consensus boundary, then immediately resumed at the start of the subsequent event. This manipulation disrupted the natural event structure by removing the critical boundary transition period. Importantly, video duration was matched between conditions to ensure that temporal length was not a confounding factor.

#### Recognition Test Stimuli

For the recognition memory test, two types of target frames were extracted from each video:

- **Before-Boundary (BB) frames:** Still frames captured just before an event boundary (the "high-stakes" temporal region where event transitions occur)
- **Event-Middle (EM) frames:** Still frames captured from the middle of an event, temporally distant from any boundary

For each target frame, a corresponding **lure frame** (novel, unseen frame with similar visual properties) was selected from the same video. Target-lure pairs were created at three difficulty levels (easy, moderate, hard) based on the degree of visual similarity between target and lure images.

---

### Design

The experiment employed a **2 x 2 mixed factorial design**:

- **Between-subjects factor:** Boundary Condition (Natural Cut [NB] vs. Abrupt Cut [AB])
- **Within-subjects factor:** Frame Type (Before-Boundary [BB] vs. Event-Middle [EM])

This design allowed us to test whether the effect of boundary disruption (AB vs. NB) differed as a function of frame position relative to event boundaries (BB vs. EM), providing a critical test of the boundary-specific encoding hypothesis.

---

### Procedure

The experiment consisted of two phases: an encoding (viewing) phase and a recognition memory test phase.

#### Phase 1: Encoding (Video Viewing)

Participants were instructed to watch a series of short video clips attentively in preparation for a later memory test. Videos were presented in randomized order. Depending on their assigned condition, participants viewed either the Natural Cut (NB) or Abrupt Cut (AB) versions of all videos.

To ensure attentive viewing, **five videos were repeated** during the encoding phase as vigilance checks. When a repeated video appeared, participants were instructed to press the spacebar to indicate they recognized it as a repeat. This vigilance task served as an attention check to verify that participants were actively encoding the video content.

The total encoding phase lasted approximately [XX minutes], with each video ranging from [X to X seconds] in duration.

#### Phase 2: Recognition Memory Test

Following the encoding phase, participants completed a **two-alternative forced-choice (2AFC) recognition memory test**. On each trial:

1. **Stimulus Presentation:** Two still frames from the same video were presented side-by-side on the screen:
   - One **target** frame (previously seen in the encoding video)
   - One **lure** frame (novel, not shown in the video, but similar in content)

2. **Recognition Response:** Participants indicated which frame was the target by pressing the corresponding key (left "L" or right "R"). Response accuracy (`resp.corr`: 0 = incorrect, 1 = correct) and response time (`resp.rt`: time in seconds from stimulus onset to keypress) were recorded.

3. **Confidence Rating:** Immediately after making their recognition decision, participants rated their confidence in their response on a **5-point Likert scale** (1 = *very unconfident*, 5 = *very confident*). Confidence ratings were recorded as `conf_radio.response`.

Each participant completed approximately **40 recognition trials** (exact number varied slightly due to experimental design), with trials blocked by video and randomized within blocks. Both BB and EM frame trials were intermixed within the recognition phase.

---

### Dependent Measures

The following primary dependent variables were analyzed:

1. **Recognition Accuracy** (`resp.corr`): Binary measure indicating correct (1) or incorrect (0) identification of the target frame. Aggregated as proportion correct per participant and per condition.

2. **Response Time (RT)** (`resp.rt`): Time in seconds from stimulus presentation to keypress response. Provides an index of retrieval speed and decision difficulty.

3. **Confidence Rating** (`conf_radio.response`): Self-reported confidence on a 1-5 scale. Used to assess metacognitive monitoring and subjective memory strength.

4. **Frame Type Effects:** Recognition performance was analyzed separately for BB (Before-Boundary) and EM (Event-Middle) frames to test boundary-specific encoding predictions.

5. **Performance Variability:** Standard deviation of accuracy across trials/videos, used to assess consistency of encoding quality.

---

### Data Preprocessing and Exclusions

#### Participant Exclusions

- The first **13 participants** were excluded from all analyses due to a data recording error that prevented proper data collection.
- This resulted in a final analytical sample of **N = 146** participants.

#### Missing Data Handling

Some participants had incomplete demographic data (age, gender, handedness, or vision status missing, *n* = 38). To preserve statistical power and avoid listwise deletion, missing values were imputed using **condition-specific measures of central tendency:**

- **Continuous variables** (age): Imputed using the median value within the participant's assigned condition (AB or NB).
- **Categorical variables** (gender, handedness, vision): Imputed using the mode (most frequent category) within the participant's condition.

Importantly, **all behavioral data** (accuracy, RT, confidence) were complete with no missing values. Imputation was applied only to demographic variables to maintain sample size for demographic descriptive statistics. To ensure transparency, binary flags (`age_imputed`, `gender_imputed`, etc.) were created to identify which participants received imputed demographic values.

#### Trial-Level Data Quality

- **Vigilance check performance:** Participants who failed to correctly identify repeated videos during the vigilance trials were flagged but retained in the primary analyses, as vigilance performance did not correlate with recognition accuracy.
- **No trial-level exclusions** were applied based on response time, as RT distributions did not reveal evidence of inattentive responding (e.g., extremely fast guessing).

---

### Analysis Approach

#### Primary Hypotheses

1. **H1 (Overall Accuracy):** NB participants will demonstrate higher overall recognition accuracy compared to AB participants, reflecting better memory encoding when event boundaries are preserved.

2. **H2 (BB-Frame Accuracy):** The NB advantage will be particularly pronounced for Before-Boundary (BB) frames, as these frames are most directly affected by boundary disruption.

3. **H3 (Confidence Ratings):** NB participants will report higher confidence in their memory judgments, reflecting stronger and more accessible memory traces.

4. **H4 (Frame Type Interaction):** A critical test of boundary-specific encoding predicts an *interaction* between Condition (AB vs. NB) and Frame Type (BB vs. EM), such that the NB advantage is larger for BB frames than for EM frames.

5. **H5 (Performance Variability):** AB participants will show greater variability in recognition performance across videos/trials, reflecting inconsistent encoding quality when boundaries are disrupted.

#### Statistical Approach

Given that normality assumptions were violated for key dependent variables (based on Shapiro-Wilk tests, Kolmogorov-Smirnov tests, and Q-Q plot inspection), **non-parametric statistical tests** were employed:

- **Between-groups comparisons:** Mann-Whitney U tests with **rank-biserial correlation** as the effect size measure.
- **Within-subjects comparisons:** Friedman tests for repeated measures.
- **Variance comparisons:** Levene's test for equality of variances.
- **Correlations:** Spearman's rank-order correlations.

**Multiple comparisons correction** was applied using the Benjamini-Hochberg False Discovery Rate (FDR) procedure to control Type I error inflation across the five primary hypothesis tests. An alpha level of .05 was used for all tests.

Descriptive statistics (medians, means, standard deviations, interquartile ranges) were computed for all dependent measures, stratified by condition and frame type. **Effect sizes** were reported for all statistical tests regardless of significance, following recommended practices for transparent reporting.

---

## DESIGN SUMMARY TABLE

| **Design Element** | **Specification** |
|-------------------|-------------------|
| **Design Type** | 2 × 2 mixed factorial |
| **Between-Subjects Factor** | Boundary Condition (NB vs. AB) |
| **Within-Subjects Factor** | Frame Type (BB vs. EM) |
| **Sample Size** | N = 146 (79 NB, 67 AB) |
| **Stimuli** | 40 YouTube Shorts with consensus boundaries |
| **Manipulation** | NB: Natural cuts; AB: Cuts 1-5s before boundary |
| **Task** | 2AFC recognition + confidence rating |
| **Trials per Participant** | ~40 recognition trials (~46.6 average) |
| **Vigilance Check** | 5 repeated videos during encoding |
| **Dependent Variables** | Accuracy, RT, Confidence, Variability |
| **Analysis** | Non-parametric tests with FDR correction |

---

## TIMELINE

| **Phase** | **Duration** | **Description** |
|-----------|-------------|-----------------|
| Instructions | ~2 min | Task explanation, consent |
| Encoding (Video Viewing) | ~[XX] min | Watch 40 video clips (NB or AB version) |
| Vigilance Trials | Intermixed | 5 repeated videos (spacebar to skip) |
| Break | ~1 min | Transition to recognition phase |
| Recognition Test | ~[XX] min | 40 trials: 2AFC + confidence rating |
| Debriefing | ~2 min | Purpose explanation |
| **Total** | **~[XX] min** | Complete experiment |

---

## COUNTERBALANCING AND CONTROLS

- **Video assignment:** Videos were randomly assigned to NB/AB conditions across participants.
- **Trial order:** Recognition test trials were randomized within participants to control for order effects.
- **Response mapping:** Left/right position of target frames was counterbalanced across trials.
- **Duration matching:** AB videos were edited to match NB video duration to eliminate confounding by video length.
- **Difficulty levels:** Target-lure difficulty (easy/moderate/hard) was balanced across conditions and frame types.
- **Frame selection:** BB and EM frames were selected systematically based on annotator consensus boundaries to ensure consistent temporal positioning.

---

## OPERATIONALIZATION OF KEY VARIABLES

### Independent Variables

**Boundary Condition** (between-subjects):
- **Natural Boundary (NB):** Videos play continuously through event boundaries
- **Abrupt Boundary (AB):** Videos cut 1-5 seconds before event boundaries

**Frame Type** (within-subjects):
- **Before-Boundary (BB):** Frames extracted from the temporal region immediately preceding an event boundary
- **Event-Middle (EM):** Frames extracted from the center of an event, temporally distant from boundaries

### Dependent Variables

**Recognition Accuracy:**
- Binary correct/incorrect on each trial
- Aggregated as proportion correct per participant
- Range: 0.0 (0% correct) to 1.0 (100% correct)

**Response Time (RT):**
- Continuous measure in seconds
- Time from frame pair presentation to keypress
- Reflects retrieval speed and confidence

**Confidence Rating:**
- Ordinal 5-point scale
- 1 = very unconfident, 5 = very confident
- Averaged across trials per participant

**Performance Variability:**
- Standard deviation of trial-level accuracy within participants
- Coefficient of variation (CV = SD/Mean × 100)
- Reflects consistency of memory encoding

---

## THEORETICAL RATIONALE

This experimental design tests the **Event Segmentation Theory** prediction that event boundaries serve as critical anchors for memory encoding. By disrupting boundaries in the AB condition while preserving them in the NB condition, we can isolate the impact of boundary continuity on recognition memory.

The inclusion of both BB (Before-Boundary) and EM (Event-Middle) frame types provides a crucial control: if boundary disruption specifically impairs boundary-related encoding, the AB condition should show particular deficits for BB frames relative to EM frames. This **interaction pattern** would support a boundary-specific mechanism rather than a general impairment.

The 2AFC recognition task with confidence ratings allows us to assess both **memory accuracy** (objective performance) and **metacognitive monitoring** (subjective confidence), providing insight into both the strength and accessibility of memory traces.

---

## STATISTICAL POWER

With N = 146 (67 AB, 79 NB), this design provides:

- **80% power** to detect a between-groups effect of **d ≈ 0.48** (medium effect) at α = .05
- **95% power** to detect a between-groups effect of **d ≈ 0.65** (medium-large effect)

For the 2 × 2 mixed ANOVA (Condition × Frame Type interaction):
- **80% power** to detect an interaction effect of **f ≈ 0.25** (medium) at α = .05

These power estimates indicate adequate sensitivity to detect theoretically meaningful effects.

---

## ETHICS AND CONSENT

[Include your institution's IRB approval statement, e.g.:]

This study was approved by the [Institution] Institutional Review Board (IRB Protocol #[XXXX]). All participants provided written informed consent prior to participation and were informed of their right to withdraw at any time without penalty. Data were anonymized and stored securely in accordance with institutional guidelines.

---

## EQUIPMENT AND SOFTWARE

- **Video presentation:** [PsychoPy version X.X / MATLAB / E-Prime, etc.]
- **Display:** [Screen size, resolution, viewing distance]
- **Response collection:** Standard keyboard (L/R keys for 2AFC, number keys for confidence)
- **Data storage:** CSV format with unique participant IDs
- **Analysis software:** Python 3.9+ (pandas, scipy, numpy, matplotlib, seaborn)

---

## DATA AVAILABILITY

[Include your data sharing statement, e.g.:]

Anonymized data, analysis scripts, and stimulus materials are available upon reasonable request / are publicly available at [repository link] / are available in accordance with institutional data sharing policies.

---

## SUMMARY OF KEY DESIGN FEATURES

✅ **Between-subjects manipulation** minimizes carryover effects  
✅ **Duration-matched conditions** eliminate confounding by video length  
✅ **Within-subjects frame type** increases statistical power  
✅ **Vigilance checks** ensure attentive encoding  
✅ **Consensus boundaries** provide objective event structure definition  
✅ **Multiple difficulty levels** minimize ceiling/floor effects  
✅ **Confidence ratings** assess metacognitive awareness  
✅ **Adequate sample size** with sufficient statistical power  
✅ **Transparent data handling** with imputation flags for missing data  

---

## HYPOTHESES MAPPED TO DESIGN

| **Hypothesis** | **Comparison** | **Prediction** |
|----------------|----------------|----------------|
| H1: Overall Accuracy | NB vs. AB (all trials) | NB > AB |
| H2: BB-Frame Accuracy | NB vs. AB (BB trials only) | NB > AB |
| H3: Confidence Ratings | NB vs. AB (mean confidence) | NB > AB |
| H4: Frame-Specific Effect | Condition × Frame Type interaction | NB advantage larger for BB than EM |
| H5: Performance Variability | SD/CV comparison (NB vs. AB) | AB > NB (more variable) |

---

**End of Methods Section**

---

## NOTES FOR YOUR REPORT

1. **Fill in the bracketed [XX] values** with actual timings from your experiment
2. **Add IRB number** and ethics approval details
3. **Specify compensation** (course credit, payment, etc.)
4. **Add equipment details** (screen size, viewing distance, software versions)
5. **Adjust language** to match your institution's style (e.g., "participants" vs "subjects")
6. **Include any additional exclusions** if you identified other data quality issues

This methods section follows APA style and provides all necessary details for replication. You can adapt sections as needed for your specific report format!
