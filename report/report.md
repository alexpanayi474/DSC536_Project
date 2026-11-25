---
title: "AI-Powered Bayesian Inference: using LLM predictions as priors for students' academic outcomes"
subtitle: "DSC536 project report, based on O'Hagan & Ročková (2025), arXiv:2502.19231"
author: "[TODO: names]"
---

> **Notes for the group (delete before submission).**
>
> * **Who writes what.** Sections 1–3 (problem, methodology, assumptions) and 10–11 (discussion,
>   conclusion) are shared: extend them for the classification part. Sections 4–8 are the regression
>   part. Section 9 is for the classification part.
> * **Length.** The brief asks for about 10–20 pages in total. The regression part is currently ~9
>   pages, so it can be shortened if needed (e.g. Section 8 or Table 2).
> * **Working together.** Always `git pull` before editing and push soon after, so we do not overwrite
>   each other. Items marked **[TODO]** need input.
> * **Numbers and figures** come from `regression/output_v3/` (main analysis, v3 LLM labels) and
>   `regression/output/` (first LLM run, v2). Keep figure paths relative to this file.

# 1. Problem and research question

The paper of O'Hagan & Ročková (2025) addresses a practical problem: labelled data are often scarce
and expensive, while generative AI models can produce predictions for many unlabelled cases at almost
no cost. These predictions are informative but cannot be trusted blindly, because the AI model may be
biased or overconfident. The question is how to use AI predictions *coherently* in a Bayesian
analysis, so that they help when they are good and do little harm when they are not.

Their answer is to treat the AI model as a source of *prior information*: AI predictions are turned
into "imaginary data" that define a prior distribution, and the strength of this prior is a tuning
parameter that can be chosen from the data.

We apply this idea to a continuous-outcome problem. Using data on university students, we ask:

* How is weekly use of AI tools (`ai_hours_per_week`) related to students' **critical-thinking score**
  (0–100) and **CGPA** (0–4)?
* If only a small number of students have known outcomes, can predictions from a large language model
  (LLM) for the remaining students improve the inference on this effect?
* How should the weight given to the LLM be chosen, and what happens when the LLM is wrong?
* **[TODO classification part: research question for the multiclass outcome.]**

This setting is interesting for two reasons. It is a natural use case of the method (outcomes such as
grades are costly to collect, but student characteristics are easy to obtain), and the topic itself
is about AI, which makes the question of trusting AI predictions particularly relevant.

# 2. Methodology

## 2.1 Target of inference

We observe $n$ labelled students $D_n = \{(Y_i, X_i)\}_{i=1}^n$ and $m$ unlabelled students
$X_1^*, \dots, X_m^*$. Following the paper, the parameter of interest is defined as the minimiser of a
loss under the unknown data-generating distribution $F_0$:
$$\theta_0 = \arg\min_\theta \int \ell(\theta; Y, X)\, dF_0(Y, X), \qquad \ell(\theta; Y, X) = (Y - X^\top\theta)^2 .$$
With squared-error loss, $\theta_0$ is the population least-squares coefficient vector. This does not
require the linear model to be exactly true: $\theta_0$ is the best linear approximation. Our quantity
of interest is the component of $\theta_0$ for `ai_hours_per_week`.

## 2.2 The AI prior: a Dirichlet process with an AI base measure

Instead of placing a prior on $\theta$, the paper places a non-parametric prior on the whole
data-generating distribution,
$$F \sim \mathrm{DP}(\alpha, F_{AI}),$$
and the prior on $\theta$ is implied through $\theta(F) = \arg\min_\theta \int \ell\, dF$. The
**base measure** $F_{AI}$ describes what the AI "believes" the data look like:

* features $X^*$ are drawn from the empirical distribution of the unlabelled students;
* given $X^*$, a label $Y^*$ is drawn from the LLM's predictive distribution for that student.

The **concentration parameter** $\alpha$ is the prior strength and can be read as an *effective
number of AI data points*. After observing $D_n$, the posterior is again a Dirichlet process,
$\mathrm{DP}(\alpha + n, G_n)$ with $G_n = \frac{\alpha}{\alpha+n}F_{AI} + \frac{1}{\alpha+n}\sum_i \delta_{(Y_i,X_i)}$:
a mixture of the real data and the AI's beliefs, weighted by $n$ and $\alpha$. Two limits are useful:
$\alpha = 0$ uses only the real data (the Bayesian bootstrap), and $\alpha \to \infty$ uses only the AI.
Treating every one of the $m$ AI-labelled rows exactly like a real row corresponds to $\alpha = m$.

## 2.3 Posterior computation: the posterior bootstrap

Samples from the posterior of $\theta$ are obtained by optimisation rather than MCMC (paper,
Algorithm 1). For $t = 1, \dots, B$:

1. draw $m$ imaginary rows $(X_j^*, Y_j^*)$ from $F_{AI}$;
2. draw weights $(w_{1:n}, w^*_{1:m}) \sim \mathrm{Dir}(1, \dots, 1, \alpha/m, \dots, \alpha/m)$;
3. compute the weighted least-squares fit
   $\theta^{(t)} = \arg\min_\theta \sum_i w_i (Y_i - X_i^\top\theta)^2 + \sum_j w_j^* (Y_j^* - X_j^{*\top}\theta)^2$.

The draws $\theta^{(1)}, \dots, \theta^{(B)}$ are independent posterior samples, so the algorithm is
simple and embarrassingly parallel. We use $B = 200$ and report the posterior mean and the 5%–95%
quantiles (90% credible interval), as in the paper.

**Intuition from the theory.** The paper's Theorem 1 shows that, with $\alpha = \gamma n$, the
posterior is approximately normal and centred at the minimiser of
$\frac{1}{n}\sum_i \ell(\theta; Y_i, X_i) + \frac{\gamma}{m}\sum_j \ell(\theta; Y_j^*, X_j^*)$,
with a sandwich covariance. The practical message is that the posterior is a compromise between the
real-data estimate and the "AI estimate", and **if the AI is biased, the posterior is biased by an
amount that grows with $\alpha$**. The method does not remove this bias; it controls its weight.

## 2.4 A parametric comparison: the power (catalytic) AI prior

The paper also discusses (Section 2.1) priors that plug AI data directly into a parametric likelihood.
We use the "population catalytic" version as a comparison:
$$\pi(\theta, \sigma^2 \mid D_n) \propto L(\theta, \sigma^2; D_n)\,
\exp\Big\{\tfrac{\alpha}{m}\sum_{j=1}^m \mathrm{E}_{AI}\big[\log L(\theta, \sigma^2; Y_j^*, X_j^*)\big]\Big\}\,\pi_0(\theta, \sigma^2),$$
where the expectation is over the LLM's predictive distribution and $\pi_0$ is a vague
normal-inverse-gamma prior. For the normal linear model this posterior is available in closed form,
because the expected log-likelihood only needs $\mathrm{E}[Y^*]$ and $\mathrm{E}[Y^{*2}]$. The paper
prefers the non-parametric version because the power prior assumes that both the model and the AI
data come from the same parametric family ("double mis-specification").

## 2.5 Choosing $\alpha$

The value of $\alpha$ decides how much the analysis relies on the AI. Following Section 4.1 of the
paper we use two data-driven rules that only use the labelled data:

* **Out-of-sample prediction (CV).** For each $\alpha$ on a grid, compute the 10-fold cross-validated
  log predictive density of the labelled data, and choose the best $\alpha$. $\alpha = 0$ means
  "the AI predictions do not add value" (paper, Section 6).
* **Coverage calibration** (paper Section 4.1.1, after Syring & Martin). Bootstrap the labelled data;
  in each bootstrap sample compute the 90% interval for every $\alpha$ and check whether it covers the
  estimate from the original labelled data (the "truth" of the bootstrap world). Choose the **largest**
  $\alpha$ whose estimated coverage is at least 90%. To keep this fast, the posterior-bootstrap
  interval inside the loop is replaced by its normal approximation from Theorem 1; we checked that the
  two agree well (`theorem1_check.csv`).

# 3. Assumptions

* **Exchangeable labelled data.** The labelled students are an i.i.d. sample from $F_0$; the
  unlabelled students come from the same population (their features define $F_{AI}$'s marginal).
* **The AI is external information.** The LLM's predictions must not be derived from the labelled
  data; otherwise the data would be used twice (the paper contrasts this with catalytic priors).
  In our implementation the LLM only sees the features of the unlabelled students.
* **Loss-based target.** Inference is about the least-squares projection $\theta_0$; the linear model
  need not be correct. (The power prior, in contrast, assumes a correct normal linear model.)
* **The AI's predictive distribution is meaningful.** The prior uses the full LLM distribution, so it
  matters whether the LLM's stated probabilities reflect real knowledge and uncertainty.
* **Choosing $\alpha$ from the labelled data is informative.** Both rules assume the labelled sample
  is large enough to tell whether the AI agrees with the data.

# 4. Data

**Source.** *Campus AI Pulse 2026: Student AI Usage & Academic Performance* (Aamir, 2026), file
`campus_ai_pulse_2026.csv`, published on Kaggle under the CC BY-NC-SA 4.0 licence. The description
states that the data are **synthetic**: "records are simulated for learning, research practice and
portfolio projects. They are not responses from real students". We therefore use the data to study how
the method behaves, not to draw conclusions about real students.

The file contains 1200 students and 24 variables on demographics, study programme, university AI
policy, AI-tool use, study habits and outcomes. We use 12 predictors: age, gender, country, university
type, programme, year of study, university AI policy, main AI tool, main use case, AI hours per week,
percentage of assignment work done with AI, and study hours per week. The other variables are the
student ID, two further indicators of AI use (`uses_ai`, which is already implied by zero AI hours,
and `paid_ai_subscription`), the self-rated prompt skill, and six variables that describe consequences of AI use measured at the
same time as the outcomes (AI-dependency score and level, time saved, integrity concern, AI-detector
flag, would recommend AI); these are not used. `primary_use_case = "No AI use"` coincides exactly with
`primary_ai_tool = "No AI tool"` (the same 148 students), so `primary_use_case` is dropped from the
models with all covariates.

**Quality checks.** The data are clean: no missing values, duplicates or impossible values, no
leakage from the outcomes (no single variable explains more than 5% of either outcome), and no rare
factor levels. Two features need care. First, consistent with the data being simulated, CGPA is
almost exactly normally distributed (skewness 0.00) and capped at 4.0. Second, the signal is weak: on
all 1200 students,

| outcome | $\hat\beta$ (AI hours) | 95% CI | $R^2$ |
|---|---|---|---|
| critical-thinking score | −0.301 | [−0.441, −0.160] | 0.015 |
| CGPA | 0.0079 | [0.0032, 0.0125] | 0.009 |

![**Figure 1.** Distributions of the two outcomes and of weekly AI hours (1200 students). CGPA is
almost exactly normal and capped at 4; 12% of students report zero AI hours.](../regression/output/data_check/distributions.png){width=100%}

One extra hour of AI use per week goes with about 0.3 points lower critical-thinking score and a
negligible change in CGPA. AI hours are strongly correlated with the share of assignments done with AI
($r = 0.87$), which inflates the uncertainty in the models with all covariates.

# 5. Eliciting the LLM prior

**Split.** With a fixed seed the students are split into 300 labelled and 900 unlabelled ("AI")
students; the outcomes of the AI students are hidden from the analysis and used only for evaluation.

**Two elicitations.** For every AI student we send the student's features to an LLM through the Groq
API and ask for a probability distribution over 8 CGPA bins (width 0.5) and 10 critical-thinking bins
(width 10), returned as JSON. The base measure $F_{AI}$ draws a bin from these probabilities and then
a value uniformly within the bin. We ran this twice:

| | first run (v2) | main run (v3), in the style of the paper |
|---|---|---|
| model | `gpt-oss-20b` | `gpt-oss-120b` (larger reasoning model; the paper used OpenAI's o4-mini) |
| students per request | 15 | 1 (as in the paper) |
| prompt | short description of the task | detailed prompt following the paper's Appendix A.1: role, meaning and scale of every feature, task, output format |
| temperature / reasoning effort | 0.4 / low | 0.7 / medium (as in the paper) |
| students labelled | 900 | 611 (limited by the free daily API quota) |

Like the paper's prompt, the v3 prompt explains the inputs in detail. Unlike the paper, it gives no
expert guidance on how each feature affects the outcomes (the paper's "disease profiles"), because
here that would mean writing our own answer into the prior.

**Lessons from implementation.** A first version of the pipeline had three problems that would have
invalidated the results: the random seed was set *after* the split, so cached LLM answers were
attached to different students on every run; answers were matched by their position in the reply
rather than by student ID; and many answers were degenerate (all probability on one bin, or the same
answer for all 15 students in a batch). These were fixed before any analysis. The method itself is
simple, but the elicitation step needs careful engineering.

**Quality of the LLM predictions.** Table 1 compares both runs with the hidden true outcomes on the
same 611 students ("LLM" = mean of the LLM's distribution for each student).

**Table 1.** LLM predictions vs true outcomes, 611 AI students.

| outcome | run | mean | correlation with truth | MSE | AI-hours slope |
|---|---|---|---|---|---|
| critical thinking | v2 | 72.0 | −0.06 | 338 | **+0.27** |
| | v3 | **62.3** | 0.06 | 170 | **−0.41** |
| | truth | 62.0 | | 145 (variance) | −0.24 |
| CGPA | v2 | 3.17 | −0.02 | 0.30 | +0.022 |
| | v3 | 2.88 | 0.07 | 0.22 | −0.007 |
| | truth | 3.06 | | 0.15 (variance) | +0.009 |

Neither run can predict individual students: the correlations are close to zero and both MSEs are
larger than the variance, i.e. worse than always predicting the average. But the two runs hold very
different *beliefs about the population*. The v3 run gets the average critical-thinking score right
and implies the right direction of the effect of AI use (though about 1.7 times too strong), whereas
v2 is ten points too optimistic and implies the **opposite** direction. For CGPA it is the other way
round: v2 has the right direction (but more than twice too strong) and v3 the wrong one. As the
results show, it is this population-level relationship, not the accuracy for individual students,
that determines whether the AI prior helps.

# 6. Experimental design

For both outcomes we fit two models: AI hours only, and AI hours plus all covariates. We use $n = 300$
and $n = 100$ labelled students and a grid of $\alpha \in \{0, 1, 3, 10, 30, 100, 300, m, 1000\}$.
We compare: real data only ($\alpha = 0$), 1:1 pooling ($\alpha = m$, i.e. treating AI labels as
real), and $\alpha$ chosen by CV or by coverage calibration, for both the posterior bootstrap and the
power prior.

The **main analysis uses the v3 labels**. On the saved split the AI data are the $m = 611$ students
labelled by v3. To evaluate the methods we repeat the analysis on **100 random splits**: in each split
519 of the 611 AI-labelled students (85%) form the AI data, the labelled students are drawn from the
remaining students, and the "truth" is the least-squares coefficient on all students *not* used as
labelled data. We report bias, root mean squared error (RMSE), coverage of the 90% interval, its
average width, and how often the estimate has the correct sign. The same design with the v2 labels
(600 of the 900 AI students per split) is summarised in Section 8.

# 7. Results

## 7.1 One split

**Table 2.** Posterior mean [90% interval] of the AI-hours coefficient on the saved split, v3 labels
($m = 611$). In brackets after the estimate: the chosen $\alpha$.

| outcome | n | truth | real data only | 1:1 pooling | power prior, $\alpha$ by CV | posterior bootstrap, $\alpha$ by calibration |
|---|---|---|---|---|---|---|
| critical thinking | 300 | −0.295 | −0.318 [−0.541, −0.094] | −0.380 [−0.560, −0.200] | −0.318 [−0.542, −0.095] (1) | −0.384 [−0.626, −0.165] (1000) |
| critical thinking | 100 | −0.305 | −0.265 [−0.607, 0.077] | −0.385 [−0.598, −0.171] | −0.269 [−0.615, 0.078] (3) | −0.394 [−0.678, −0.140] (1000) |
| CGPA | 300 | 0.0083 | 0.0065 [−0.0011, 0.0141] | −0.0026 [−0.0101, 0.0049] | 0.0065 [−0.0011, 0.0141] (0) | 0.0035 [−0.0053, 0.0130] (100) |
| CGPA | 100 | 0.0077 | 0.0090 [−0.0041, 0.0221] | −0.0047 [−0.0138, 0.0044] | 0.0088 [−0.0044, 0.0220] (1) | 0.0019 [−0.0112, 0.0161] (100) |

![**Figure 2.** Posterior mean and 90% interval of the AI-hours coefficient as a function of $\alpha$
(the effective number of AI data points), 100 labelled students, saved split, v3 labels. Blue: power
prior; orange: posterior bootstrap (Dirichlet-process prior). Dashed line: truth. The star and the
triangles mark the $\alpha$ chosen by cross-validation and by coverage calibration.](../regression/output_v3/beta_ai_vs_alpha_n100.png){width=100%}

For critical thinking, adding the AI prior moves the estimate towards the LLM's slope, which points
in the right direction. With 100 labelled students the real data alone are inconclusive (the interval
contains zero), while 1:1 pooling gives −0.385 [−0.598, −0.171]: a clearly negative effect, with an
interval that still contains the truth (−0.305). For CGPA the LLM's slope has the wrong sign, and
1:1 pooling pushes the estimate below zero, so that its interval no longer contains the truth. CV
gives the AI rows almost no weight in all four cases; coverage calibration gives them a large weight.

## 7.2 100 random splits

**Table 3.** Performance over 100 random splits, AI-hours coefficient, v3 labels (90% intervals; 1:1
pooling and CV use the power prior).

| outcome, model | n | method | mean $\alpha$ | bias | RMSE | coverage | width | correct sign |
|---|---|---|---|---|---|---|---|---|
| critical thinking, AI hours only | 300 | real data only | 0 | −0.062 | 0.160 | 0.85 | 0.46 | 1.00 |
| | | 1:1 pooling | 519 | −0.101 | 0.126 | 0.89 | 0.38 | 1.00 |
| | | CV | 4.7 | −0.063 | 0.158 | 0.85 | 0.47 | 1.00 |
| | | posterior bootstrap, calibration | 861 | −0.092 | 0.134 | 0.98 | 0.52 | 1.00 |
| critical thinking, AI hours only | 100 | real data only | 0 | −0.036 | 0.237 | 0.89 | 0.81 | 0.92 |
| | | 1:1 pooling | 519 | −0.102 | **0.113** | 1.00 | **0.46** | 1.00 |
| | | CV | 5.0 | −0.041 | 0.227 | 0.92 | 0.81 | 0.93 |
| | | posterior bootstrap, calibration | 653 | −0.077 | 0.150 | 0.99 | 0.67 | 0.98 |
| critical thinking, all covariates | 100 | real data only | 0 | 0.089 | 0.597 | 0.81 | 1.63 | 0.76 |
| | | 1:1 pooling | 519 | 0.363 | 0.384 | 0.82 | 0.99 | 0.88 |
| | | CV | 111 | 0.215 | 0.422 | 0.92 | 1.59 | 0.82 |
| CGPA, AI hours only | 300 | real data only | 0 | −0.0011 | 0.0052 | 0.89 | 0.016 | 0.97 |
| | | 1:1 pooling | 519 | −0.0099 | 0.0102 | **0.21** | 0.016 | **0.08** |
| | | CV | 1.8 | −0.0012 | 0.0051 | 0.90 | 0.016 | 0.97 |
| | | posterior bootstrap, calibration | 213 | −0.0052 | 0.0077 | 0.77 | 0.020 | 0.67 |
| CGPA, AI hours only | 100 | real data only | 0 | 0.0001 | 0.0088 | 0.90 | 0.028 | 0.85 |
| | | 1:1 pooling | 519 | −0.0123 | 0.0124 | **0.07** | 0.020 | **0.00** |
| | | CV | 1.7 | −0.0002 | 0.0086 | 0.90 | 0.028 | 0.84 |
| | | posterior bootstrap, calibration | 329 | −0.0066 | 0.0107 | 0.76 | 0.030 | 0.48 |

The complete tables are in `regression/output_v3/`. The main findings are:

1. **When the LLM's belief points the right way, the AI prior helps, most of all with few labels.**
   For critical thinking with 100 labelled students, 1:1 pooling halves the RMSE (0.237 → 0.113),
   almost halves the width of the interval (0.81 → 0.46), and gives the correct sign in every split.
   With 300 labelled students the gain is smaller (RMSE 0.160 → 0.126), as expected when the real
   data are already informative. The estimates are slightly too negative (bias −0.10) because the
   LLM exaggerates the effect, but the intervals still contain the truth in 89% (300 labelled) and
   100% (100 labelled) of splits. The RMSE falls steadily as $\alpha$ grows (Figure 4).
2. **When the LLM's belief points the wrong way, a large AI weight fails.** For CGPA, 1:1 pooling
   covers the truth in only 21% (300 labelled) and 7% (100 labelled) of splits and gives the wrong
   sign almost always. A moderate weight still helps slightly (with 100 labelled students the RMSE
   falls from 0.0088 at $\alpha = 0$ to 0.0072 at $\alpha = 30$), because pulling a noisy estimate
   towards a near-zero value reduces its variance more than it adds bias; beyond $\alpha \approx 100$
   the coverage collapses.
3. **CV is safe but too cautious.** CV gives the AI rows a weight of about 2–5 data points in the
   AI-hours-only models. This protects the CGPA analysis (coverage 0.90), but it also misses most of
   the benefit for critical thinking (RMSE 0.227 instead of 0.113 with 100 labelled students). The
   reason is that CV judges the AI by how well it predicts *individual* students, which the LLM cannot
   do (Table 1), while the useful information is the population-level relationship.
4. **Coverage calibration captures the benefit but not the risk.** It gives the AI a large weight in
   both cases: for critical thinking this works well (RMSE 0.150, coverage 0.99 with 100 labelled
   students), but for CGPA it lets the LLM's bias through (coverage 0.76, correct sign in 48% of
   splits). Its bootstrap estimate of coverage follows the actual coverage only roughly (Figure 3).
5. **The posterior bootstrap gives more honest intervals than the power prior.** The two give almost
   the same estimates, but with a large AI weight the posterior bootstrap's intervals are wider and
   cover the truth more often: 0.98 vs 0.76 for critical thinking with calibration (300 labelled), and
   0.63 vs 0.21 for CGPA with 1:1 pooling (300 labelled). This supports the paper's argument for the
   non-parametric prior.
6. **With many covariates, a moderate AI weight is best.** For critical thinking with all covariates
   and 100 labelled students, the RMSE falls from 0.597 to 0.367 at $\alpha = 100$ and rises again for
   larger $\alpha$ (Figure 4); CV chooses $\alpha \approx 111$ on average and reaches coverage 0.92.

![**Figure 3.** Coverage of the 90% intervals over 100 random splits (100 labelled students, v3 labels)
as a function of $\alpha$. Solid: actual coverage (using the truth); dashed: bootstrap estimate used by
the calibration rule (labelled data only). Dotted line: nominal 90%.](../regression/output_v3/calibration_n100.png){width=100%}

![**Figure 4.** RMSE of the AI-hours coefficient over 100 random splits as a function of $\alpha$, v3
labels, for the power prior (blue) and the posterior bootstrap (orange), with 300 (solid) and 100
(dashed) labelled students.](../regression/output_v3/replicates_rmse_by_alpha.png){width=100%}

# 8. Comparison with the first LLM run (v2)

The same analysis with the v2 labels shows what happens when the LLM's belief is wrong for the main
outcome. v2 implies that more AI use goes with *higher* critical thinking (Table 1), and Table 4 shows
the consequence.

**Table 4.** Critical thinking, AI hours only, 100 random splits, v2 labels (600 of 900 AI students
per split).

| n | method | mean $\alpha$ | bias | RMSE | coverage | correct sign |
|---|---|---|---|---|---|---|
| 300 | real data only | 0 | 0.012 | 0.146 | 0.91 | 0.99 |
| | 1:1 pooling | 600 | 0.432 | 0.442 | **0.01** | **0.02** |
| | CV | 1.3 | 0.015 | 0.146 | 0.91 | 0.99 |
| | posterior bootstrap, calibration | 53 | 0.095 | 0.191 | 0.86 | 0.95 |
| 100 | real data only | 0 | 0.052 | 0.293 | 0.85 | 0.82 |
| | 1:1 pooling | 600 | 0.557 | 0.563 | **0.00** | **0.00** |
| | CV | 1.3 | 0.060 | 0.292 | 0.86 | 0.80 |
| | posterior bootstrap, calibration | 180 | 0.231 | 0.377 | 0.73 | 0.58 |

With v2, treating the AI labels as real data turns the negative effect into a positive one in almost
every split (Figure 5). CV again gives the AI almost no weight and protects the analysis, while
coverage calibration lets part of the bias through when only 100 labelled students are available.
(The rows for "real data only" differ slightly from Table 3 because the random splits differ.)

![**Figure 5.** As Figure 2, but with the v2 labels and 300 labelled students: with $\alpha = m$ the
estimate becomes positive.](../regression/output/beta_ai_vs_alpha_n300.png){width=100%}

The same data, the same method and the same family of models therefore lead to opposite conclusions
depending on the prompt and the LLM: the v3 prior sharpens the inference on critical thinking, while
the v2 prior reverses it.

# 9. Application to a multiclass outcome

**[TODO – classification part.]** Suggested structure, mirroring Sections 4–8: the outcome and its
classes, how the LLM probabilities were elicited, the model and loss (e.g. multinomial logistic
regression with cross-entropy loss in the posterior bootstrap), how $\alpha$ was chosen, results, and
the main findings. Where possible use the same split, seed and $\alpha$ grid as the regression part, so
the two parts can be compared in the discussion.

# 10. Discussion

**Advantages.** The method is simple to implement (a weighted least-squares fit repeated $B$ times),
does not require a correct parametric model, gives fully Bayesian uncertainty quantification, and has a
single interpretable tuning parameter: $\alpha$, the number of data points the AI is worth. It uses the
AI's full predictive distribution rather than a single predicted value. In our application a good AI
prior halved the error of the estimated effect when only 100 labelled students were available.

**What the AI prior contributes.** Neither LLM could predict individual students, yet one of them
improved the inference. For a regression coefficient, what matters is whether the relationship implied
by the LLM's predictions is close to the true one. A plausible reason why the v3 LLM got the direction
right for critical thinking is that the synthetic data were generated under the same common belief
that heavy AI use harms critical thinking; on synthetic data an LLM can only contribute such general
beliefs, not real knowledge about the students.

**Limitations and failure modes observed in our application.**

* *Bias is not corrected, only weighted.* When the AI's belief is wrong (v2 for critical thinking,
  v3 for CGPA), any large $\alpha$ moves the posterior towards the wrong answer (Theorem 1), up to
  reversing the sign of the effect.
* *Choosing $\alpha$ involves a trade-off.* CV was always safe but missed most of the benefit, because
  it measures predictive accuracy for individuals rather than the quality of the implied relationship.
  Coverage calibration captured the benefit when the LLM was right but let its bias through when it was
  wrong, especially with few labelled students. Neither rule was both safe and efficient, so results
  should be reported over a range of $\alpha$.
* *Dependence on the elicitation.* Two prompts and models gave opposite priors for the same question,
  and a first implementation contained errors that would have invalidated the results.
* *Practical constraints.* The paper-style elicitation (one student per request) used the free API
  quota quickly, so the v3 prior is based on 611 rather than 900 students.
* *Synthetic data.* The estimated effects describe the simulated data, not real students.

**Practical recommendations.** Never treat AI labels as real data without checking them; compare the
relationship implied by the AI's predictions with the labelled data before trusting it; report results
for a range of $\alpha$, not only for one selected value; prefer the non-parametric posterior bootstrap,
whose intervals stayed more honest under heavy AI weight; and be cautious with coverage calibration
when the labelled sample is small.

# 11. Conclusion

We applied the AI-powered Bayesian inference framework to estimate the effect of AI use on students'
critical thinking and CGPA, using an LLM to build the prior. With an LLM prompted in the style of the
paper, the AI prior pointed in the right direction for critical thinking and, with only 100 labelled
students, halved the error of the estimated effect while keeping valid coverage. For CGPA, and for
critical thinking with a weaker first prompt, the LLM's belief pointed the wrong way, and giving it a
large weight produced confidently wrong conclusions. Cross-validation protected against this but was
too cautious to exploit a good prior, while coverage calibration did the opposite. The application
shows both the main strength of the approach, an explicit and tunable trust in the AI, and its main
limitation: how much to trust the AI must be learned from the labelled data, and this is hard when
labelled data are scarce.

# Use of AI tools

**[Agree the wording with the group.]** AI tools were used to help understand the paper and to write
and debug parts of the R code. All analysis choices, results and interpretations were checked by the
authors.

# References

* O'Hagan, S. and Ročková, V. (2025). AI-Powered Bayesian Inference. arXiv:2502.19231.
* Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I. and Zrnic, T. (2023).
  Prediction-powered inference. *Science* 382(6671), 669–674.
* Syring, N. and Martin, R. (2019). Calibrating general posterior credible regions. *Biometrika*
  106(2), 479–486.
* Ferguson, T. S. (1973). A Bayesian analysis of some nonparametric problems. *Annals of Statistics*
  1(2), 209–230.
* Aamir, M. (2026). *Campus AI Pulse 2026: Student AI Usage & Academic Performance* [synthetic
  dataset]. Kaggle. https://www.kaggle.com/datasets/aamir28/student-ai-usage-and-academic-performance-2026
  (CC BY-NC-SA 4.0).
