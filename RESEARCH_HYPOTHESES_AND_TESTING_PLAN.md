# BRSM Movie Memory - Research Hypotheses & Statistical Testing Plan

## EVALUATION OF YOUR CURRENT HYPOTHESES

### ✅ **Hypothesis 1: Overall Recognition Accuracy**
**H0:** No difference in overall recognition accuracy between NB and AB groups  
**H1:** NB participants show higher overall recognition accuracy than AB participants

**Status:** ✅ **EXCELLENT** - This is your primary hypothesis  
**Preliminary Data:** NB (87.2%) vs AB (83.8%) - 3.4% difference  
**Recommended Test:** Independent samples t-test (if normal) or Mann-Whitney U test  
**Effect Size:** Cohen's d  
**Why this matters:** Tests the core prediction that event boundaries facilitate memory encoding

---

### ✅ **Hypothesis 2: BB-Frame Recognition Accuracy**
**H0:** No difference in BB-frame accuracy between NB and AB groups  
**H1:** NB participants show higher BB-frame accuracy (boundary-anchored encoding)

**Status:** ✅ **EXCELLENT** - Tests the specific mechanism  
**Recommended Test:** Independent samples t-test or Mann-Whitney U test  
**Additional Analysis:** Mixed ANOVA with Frame Type (BB vs EM) × Condition (AB vs NB)  
**Why this matters:** BB frames should show boundary-specific effects

---

### ✅ **Hypothesis 3: Overall Confidence Ratings**
**H0:** No difference in mean confidence ratings between NB and AB groups  
**H1:** NB participants report higher confidence ratings

**Status:** ✅ **GOOD**  
**Preliminary Data:** NB (4.209) vs AB (4.077) - 0.132 difference  
**Recommended Test:** Independent samples t-test or Mann-Whitney U test  
**Why this matters:** Confidence reflects memory trace strength

---

### ✅ **Hypothesis 4: BB-Frame Confidence (with EM control)**
**H0:** No difference in confidence for BB frames between NB and AB groups  
**H1:** NB participants report higher confidence for BB frames (EM frames as control)

**Status:** ✅ **EXCELLENT** - Has built-in control  
**Recommended Test:** Mixed ANOVA: 2 (Condition: AB vs NB) × 2 (Frame Type: BB vs EM)  
**Predicted Pattern:** Interaction effect - difference should be larger for BB than EM  
**Why this matters:** Shows boundary-specificity, not just general superiority

---

### ✅ **Hypothesis 5: Accuracy Variability Across Movies**
**H0:** No difference in accuracy variability (SD) across movies between AB and NB groups  
**H1:** AB participants show higher variability (differential disruption severity)

**Status:** ✅ **VERY CREATIVE** - Tests consistency of effect  
**Recommended Test:** Levene's test for equality of variances  
**Additional Analysis:** Calculate within-participant SD across movies, then compare groups  
**Why this matters:** Tests whether abrupt cuts affect movies differentially

---

## ADDITIONAL RESEARCH QUESTIONS TO CONSIDER

### **RQ6: Response Time Differences** ⭐ RECOMMENDED
**H0:** No difference in mean response time between NB and AB groups  
**H1:** AB participants show longer/shorter response times

**Rationale:** 
- Longer RT might indicate retrieval difficulty (weaker memory traces)
- Shorter RT might indicate less deliberation (lower confidence)

**Your Data:** AB (5.582s) vs NB (5.834s) - NB is actually slower!  
**Interpretation:** Slower RT with higher accuracy might reflect more careful retrieval

**Recommended Tests:**
- Independent t-test for overall RT comparison
- Examine RT for correct vs incorrect trials separately
- RT × Accuracy interaction

---

### **RQ7: Confidence-Accuracy Relationship** ⭐⭐ HIGHLY RECOMMENDED
**H0:** No difference in confidence-accuracy relationship between conditions  
**H1:** NB group shows stronger confidence-accuracy calibration

**Rationale:** Better memory should produce better metacognitive monitoring

**Recommended Tests:**
- Gamma correlation between confidence and accuracy (by participant, by condition)
- Calibration curves (% correct at each confidence level)
- Compare gamma correlations between groups (Fisher's z-test)

**Why this matters:** Shows whether boundary preservation affects metacognition

---

### **RQ8: Target vs Lure Performance** ⭐ RECOMMENDED
**H0:** No condition difference in target vs lure discrimination  
**H1:** AB shows differential impairment for targets vs lures

**Rationale:** Abrupt cuts might specifically impair encoding of targets or increase false recognition of lures

**Recommended Tests:**
- 2 (Condition) × 2 (Stimulus Type: Target vs Lure) mixed ANOVA on accuracy
- Signal detection theory: Calculate d' (sensitivity) and c (response bias)

**Why this matters:** Distinguishes encoding deficit from response bias

---

### **RQ9: Frame Position Effects** ⭐⭐ HIGHLY RECOMMENDED
**H0:** No difference in how BB vs EM frames are affected by condition  
**H1:** Condition effect is larger for BB frames than EM frames (interaction)

**Rationale:** This is the KEY mechanistic test of your theory

**Recommended Tests:**
- **2 × 2 Mixed ANOVA:** Condition (between) × Frame Type (within)
- **Primary prediction:** Significant interaction
- **Post-hoc:** Simple effects analysis

**Expected Pattern:**
- BB frames: NB > AB (large effect)
- EM frames: NB ≈ AB (small or no effect)

**Why this matters:** This is THE CRITICAL TEST of boundary-specific disruption

---

### **RQ10: Serial Position Effects** ⭐ RECOMMENDED
**H0:** No difference in early vs late trial performance between conditions  
**H1:** Condition effects differ across time (fatigue/practice interactions)

**Rationale:** Controls for non-specific effects

**Recommended Tests:**
- Split trials into quartiles (1-10, 11-20, 21-30, 31-40)
- 2 (Condition) × 4 (Quartile) mixed ANOVA
- Linear trend analysis

---

### **RQ11: Individual Differences - Age Effects** (if you have enough variance)
**Rationale:** Older participants might be more affected by disruption

**Recommended Tests:**
- Regression: Accuracy ~ Condition + Age + Condition×Age
- Median split analysis if age relationships are nonlinear

---

### **RQ12: Gender Differences** (exploratory)
**Rationale:** Check if effects generalize across gender

**Recommended Tests:**
- 2 (Condition) × 2 (Gender) ANOVA
- Primary interest: No Gender × Condition interaction (effect is general)

---

## STATISTICAL TEST SELECTION GUIDE

### **Step 1: Check Assumptions**

Before any test, check:
1. **Normality** (already planned in your visualization script)
   - Shapiro-Wilk test
   - Q-Q plots
   - If p > 0.05: Normal, use parametric tests
   - If p < 0.05: Non-normal, use non-parametric tests

2. **Homogeneity of Variance** (for t-tests and ANOVAs)
   - Levene's test
   - If violated: Use Welch's t-test or robust ANOVA

3. **Independence of Observations**
   - Between-subjects comparisons satisfy this
   - Within-subjects need repeated measures ANOVA

---

### **Test Selection Flowchart**

#### **Comparing 2 Groups (AB vs NB) on 1 Variable**

**Parametric (if assumptions met):**
- ✅ **Independent samples t-test**
- Report: t-statistic, df, p-value, Cohen's d
- Example: Overall accuracy, overall RT, overall confidence

**Non-parametric (if assumptions violated):**
- ✅ **Mann-Whitney U test**
- Report: U-statistic, Z-score, p-value, rank-biserial correlation
- Example: If accuracy is not normally distributed

---

#### **Comparing 2 Groups Across Multiple Levels**

**Example:** Condition (AB vs NB) × Frame Type (BB vs EM)

**Parametric:**
- ✅ **Mixed ANOVA** (2-way)
  - Between-subjects factor: Condition
  - Within-subjects factor: Frame Type
  - Report: F-statistic, df, p-value, η² (eta-squared)
  - **Key interest:** Interaction effect

**Non-parametric:**
- ✅ **Aligned Rank Transform ANOVA** (if assumptions violated)
- ✅ **Friedman test + Mann-Whitney** (alternative approach)

---

#### **Examining Relationships Between Variables**

**Continuous × Continuous:**
- Parametric: **Pearson correlation** (r)
- Non-parametric: **Spearman correlation** (ρ)
- Example: Age × Accuracy, RT × Confidence

**Categorical × Continuous:**
- **Point-biserial correlation**
- Example: Condition (0/1) × Accuracy

---

#### **Signal Detection Theory Analysis**

For Target/Lure discrimination:
- **d' (d-prime):** Sensitivity index
  - d' = Z(Hit Rate) - Z(False Alarm Rate)
- **c (criterion):** Response bias
  - c = -0.5 × [Z(Hit Rate) + Z(False Alarm Rate)]

Compare d' and c between conditions using t-tests

---

## COMPREHENSIVE TESTING PROCEDURE

### **Phase 1: Assumption Checking** ✅ (Already in your visualization script)

Run your normality tests and check:
- [ ] Shapiro-Wilk results
- [ ] Q-Q plot visual inspection
- [ ] Levene's test for variance homogeneity

**Decision:** Based on results, choose parametric vs non-parametric

---

### **Phase 2: Primary Hypotheses (Your Current RQs)**

#### **Test 1: Overall Accuracy (H1)**
```
Test: Independent t-test (or Mann-Whitney U)
Variables: DV = accuracy, IV = condition
Expected: NB > AB
Significance: α = 0.05
Effect size: Cohen's d (small: 0.2, medium: 0.5, large: 0.8)
```

#### **Test 2: BB-Frame Accuracy (H2)**
```
Test: Independent t-test on BB trials only
Variables: DV = BB_accuracy, IV = condition
Expected: NB > AB
```

#### **Test 3: Overall Confidence (H3)**
```
Test: Independent t-test
Variables: DV = mean_confidence, IV = condition
Expected: NB > AB
```

#### **Test 4: Frame-Specific Effects (H4 - CRITICAL TEST)**
```
Test: 2 × 2 Mixed ANOVA
Design:
  - Between: Condition (AB vs NB)
  - Within: Frame Type (BB vs EM)
  - DV: Accuracy OR Confidence

Expected pattern:
  - Main effect of Condition: NB > AB
  - Main effect of Frame Type: (maybe) BB > EM
  - INTERACTION: Condition effect larger for BB than EM

Post-hoc:
  - Simple effects: Test condition at each frame type
  - Pairwise comparisons with Bonferroni correction
```

#### **Test 5: Variability (H5)**
```
Test A: Levene's test for equality of variances
Variables: Accuracy variance by condition

Test B: 
  1. Calculate SD across movies for each participant
  2. Compare participant-level SDs between conditions (t-test)
Expected: AB > NB (more variable)
```

---

### **Phase 3: Extended Analyses (Additional RQs)**

#### **Test 6: Response Time**
```
Test: Independent t-test
Variables: DV = mean_rt, IV = condition
Interpretation: Consider RT-accuracy tradeoff
```

#### **Test 7: Confidence-Accuracy Calibration**
```
Test A: Gamma correlation (within-participant)
  - Calculate gamma for each participant
  - Compare gammas between conditions (t-test)

Test B: Calibration analysis
  - For each confidence level: compute % correct
  - Plot calibration curves
  - Compare slopes/intercepts
```

#### **Test 8: Signal Detection Theory**
```
Step 1: Calculate for each participant
  - Hit Rate (correct "R" for targets)
  - False Alarm Rate (incorrect "R" for lures)
  - d' = Z(HR) - Z(FAR)
  - c = -0.5 × [Z(HR) + Z(FAR)]

Step 2: Compare d' and c between conditions
  - t-test on d' (sensitivity)
  - t-test on c (bias)

Expected: NB shows higher d' (better discrimination)
```

#### **Test 9: Condition × Frame Type Interaction (Mechanistic Test)**
```
This is essentially Test 4 above - your most important analysis
Run for both accuracy and confidence as DVs
```

---

### **Phase 4: Control Analyses**

#### **Test 10: Serial Position**
```
Test: 2 × 4 Mixed ANOVA
Design:
  - Between: Condition
  - Within: Trial Quartile (1-10, 11-20, 21-30, 31-40)
Expected: No interaction (effects are stable across time)
```

#### **Test 11: Demographics Control**
```
Test A: Age effects
  - t-test: Compare age between conditions (should be ns)
  - Regression: Accuracy ~ Condition + Age + Condition×Age

Test B: Gender effects
  - Chi-square: Gender distribution by condition (should be ns)
  - 2×2 ANOVA: Condition × Gender (expect no interaction)

Test C: Vision effects
  - Chi-square: Vision distribution by condition
  - Compare with/without vision correction
```

---

## MULTIPLE COMPARISONS CORRECTION

**Problem:** Running many tests increases Type I error (false positives)

**Solutions:**

1. **Bonferroni Correction** (conservative)
   - Adjusted α = 0.05 / number of tests
   - Example: 10 tests → α = 0.005

2. **Holm-Bonferroni** (less conservative)
   - Rank p-values, apply sequential correction

3. **False Discovery Rate (FDR)** (recommended for exploratory)
   - Benjamini-Hochberg procedure
   - Controls proportion of false discoveries

4. **Family-Wise Approach** (recommended)
   - Core hypotheses (H1-H5): α = 0.05 (confirmatory)
   - Additional analyses: α = 0.01 (exploratory)
   - Clearly label confirmatory vs exploratory

**Recommendation:** 
- Use α = 0.05 for your 5 primary hypotheses (H1-H5)
- Use α = 0.01 for additional exploratory analyses (RQ6-12)
- Report both corrected and uncorrected p-values
- Report effect sizes regardless of significance

---

## EFFECT SIZE REPORTING

**Always report effect sizes!** Statistical significance depends on sample size.

### **For t-tests:**
- **Cohen's d:** (M₁ - M₂) / Pooled SD
  - Small: 0.2
  - Medium: 0.5
  - Large: 0.8

### **For ANOVA:**
- **η² (eta-squared):** SS_effect / SS_total
- **ηp² (partial eta-squared):** SS_effect / (SS_effect + SS_error)
  - Small: 0.01
  - Medium: 0.06
  - Large: 0.14

### **For correlations:**
- **Pearson r or Spearman ρ:**
  - Small: 0.1
  - Medium: 0.3
  - Large: 0.5

### **For non-parametric tests:**
- **Rank-biserial correlation** (Mann-Whitney U)
- **Epsilon-squared** (Kruskal-Wallis)

---

## REPORTING TEMPLATE

### **Example for Test 1 (Overall Accuracy):**

> **Hypothesis 1:** We predicted that participants viewing naturally-cut videos (NB condition) would show higher overall recognition accuracy than those viewing abruptly-cut videos (AB condition).
>
> **Analysis:** An independent samples t-test [or Mann-Whitney U test if assumptions violated] compared mean accuracy scores between conditions.
>
> **Results:** [IF PARAMETRIC] Accuracy was significantly higher in the NB condition (M = 0.872, SD = 0.069) compared to the AB condition (M = 0.838, SD = 0.084), t(144) = [X.XX], p = [.XXX], Cohen's d = [X.XX], 95% CI [X.XX, X.XX].
>
> [IF NON-PARAMETRIC] Mann-Whitney U test revealed significantly higher accuracy in the NB condition (Mdn = 0.875) compared to AB (Mdn = 0.850), U = [XXX], Z = [X.XX], p = [.XXX], rank-biserial r = [X.XX].
>
> **Interpretation:** Natural event boundaries facilitated recognition memory, with a [small/medium/large] effect size, supporting our hypothesis that preserved event structure enhances memory encoding.

---

## PRIORITY ORDER FOR ANALYSIS

### **MUST DO (Core thesis)**
1. ✅ Test 4: **Condition × Frame Type ANOVA** (THE CRITICAL TEST)
2. ✅ Test 1: Overall Accuracy comparison
3. ✅ Test 2: BB-Frame specific accuracy
4. ✅ Test 3: Overall Confidence comparison

### **SHOULD DO (Strengthens argument)**
5. ✅ Test 8: Signal Detection Theory (d' and c)
6. ✅ Test 7: Confidence-Accuracy calibration
7. ✅ Test 5: Variability analysis
8. ✅ Test 6: Response Time analysis

### **NICE TO HAVE (Controls and exploration)**
9. ⚪ Test 10: Serial position controls
10. ⚪ Test 11: Demographics controls
11. ⚪ Additional exploratory analyses

---

## POWER ANALYSIS CONSIDERATIONS

**Current Sample:**
- N = 146 total
- AB: n = 67
- NB: n = 79

**Power for independent t-test:**
- To detect medium effect (d = 0.5) at α = 0.05: Need n ≈ 64 per group
- **You have sufficient power!** ✅

**Power for 2×2 Mixed ANOVA (interaction):**
- Typically needs larger samples for adequate power
- Consider effect size from pilot data if available
- May need f = 0.25 (medium) to detect reliably

**Recommendation:** You have good power for main effects, acceptable power for interactions

---

## NEXT STEPS - ACTION PLAN

### **Immediate (Now):**
1. ✅ Run normality tests (use your visualization script)
2. ✅ Review Q-Q plots and decide parametric vs non-parametric
3. ✅ Create analysis script for primary hypotheses (H1-H5)

### **Week 1:**
4. ⚪ Run Tests 1-5 (your primary hypotheses)
5. ⚪ Calculate effect sizes
6. ⚪ Create results tables and figures

### **Week 2:**
7. ⚪ Run Tests 6-9 (extended analyses)
8. ⚪ Run Signal Detection Theory analysis
9. ⚪ Run Confidence-Accuracy calibration

### **Week 3:**
10. ⚪ Run control analyses (demographics, serial position)
11. ⚪ Apply multiple comparisons corrections
12. ⚪ Finalize results tables

### **Week 4:**
13. ⚪ Create publication-quality figures
14. ⚪ Write results section
15. ⚪ Interpret findings in discussion

---

## SOFTWARE RECOMMENDATIONS

**For Python (recommended - you're already using it):**
- `scipy.stats`: t-tests, Mann-Whitney, ANOVA, correlations
- `statsmodels`: Mixed models, advanced ANOVA
- `pingouin`: Effect sizes, pairwise comparisons, ANOVAs
- `pandas`: Data manipulation
- `matplotlib/seaborn`: Visualization

**For R (alternative):**
- `lme4`: Mixed models
- `afex`: ANOVA with effect sizes
- `emmeans`: Post-hoc comparisons
- `psycho`: Signal detection theory

---

## FINAL RECOMMENDATIONS

### **Your Hypotheses:**
✅ All 5 are excellent and theoretically motivated  
✅ H4 (Frame Type × Condition interaction) is your strongest test  
✅ H5 (variability) is creative and adds depth  

### **Must-Add Analyses:**
⭐⭐ **RQ9: Frame Type × Condition ANOVA** (if not already H4)  
⭐⭐ **RQ7: Confidence-Accuracy calibration**  
⭐ **RQ8: Signal Detection Theory (d' and c)**  

### **Testing Strategy:**
1. Check assumptions first (normality, homogeneity)
2. Run primary hypotheses with correction
3. Add extended analyses as exploratory
4. Always report effect sizes
5. Create clear visualizations

### **Expected Timeline:**
- Analysis: 2-3 weeks
- Writing results: 1-2 weeks
- Total: 3-5 weeks to complete statistical testing

---

## QUESTIONS TO CONSIDER

1. **Do you have BB vs EM frame information encoded in your data?**
   - Need to extract frame type from image filenames (I see BB and EM in paths)

2. **Do you have movie-level identifiers?**
   - Needed for variability analysis (H5)

3. **Do you want to analyze by trial type explicitly?**
   - Some analyses require separating targets from lures

4. **What is your primary research question?**
   - This determines which test is MOST critical
   - I suggest: Frame Type × Condition interaction for accuracy

---

**Ready to proceed? I can create the analysis script for any of these tests!**
