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
> * **Length.** The brief asks for about 10–20 pages in total. The regression part is currently ~7–8
>   pages, so it can be shortened if needed.
> * **Working together.** Always `git pull` before editing and push soon after, so we do not overwrite
>   each other. Items marked **[TODO]** need input.
> * **Figures** are taken from the analysis folders (e.g. `regression/output/`); keep paths relative to
>   this file. Regression numbers currently come from the v2 LLM labels; they will be updated with v3.

# 1. Problem and research question

The paper of O'Hagan & Ročková (2025) addresses a practical problem: labelled data are often scarce
and expensive, while generative AI models can produce predictions for many unlabelled cases at almost
no cost. These predictions are informative but cannot be trusted blindly, because the AI model may be
biased or overconfident. The question is how to use AI predictions *coherently* in a Bayesian
analysis, so that they help when they are good and do little harm when they are not.

Their answer is to treat the AI model as a source of *prior information*: AI predictions are turned
into "imaginary data" that define a prior distribution, and the strength of this prior is a tuning
parameter that can be chosen from the data.

We apply this idea to a continuous-outcome problem. Using survey data on university students, we ask:

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
simple and embarrassingly parallel. We report the posterior mean and the 5%–95% quantiles (90%
credible interval), as in the paper.

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
  two agree closely (`theorem1_check.csv`).

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

**Source.** `campus_ai_pulse_2026.csv` **[TODO: source/link; state whether it is synthetic]**.
It contains 1200 students and 24 variables on demographics, study programme, university AI policy,
AI-tool use, study hours, and outcomes. We use 12 predictors: age, gender, country, university type,
programme, year of study, university AI policy, main AI tool, main use case, AI hours per week,
percentage of assignment work done with AI, and study hours per week. Variables that are other
outcomes or derived scores (e.g. AI-dependency score, time saved, detector flags) and the ID are
removed. `primary_use_case = "No AI use"` coincides exactly with `primary_ai_tool = "No AI tool"`
(the same 148 students), so `primary_use_case` is dropped from the models with all covariates.

**Quality checks.** The data are clean: no missing values, duplicates or impossible values, no
leakage from the outcomes, and no rare factor levels. Two features need care. First, CGPA is almost
exactly normally distributed (skewness 0.00) and capped at 4.0, which together with the name of the
file suggests that the data are **simulated**. Second, the signal is weak: on all 1200 students,

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

**Prompt and model.** For every AI student we send the student's features to an LLM
(`gpt-oss-20b` via the Groq API, temperature 0.4) and ask for a probability distribution over 8 CGPA
bins (width 0.5) and 10 critical-thinking bins (width 10), returned as JSON. Students are sent in
batches of 15 and matched back by their ID. The base measure $F_{AI}$ draws a bin from the LLM's
probabilities and then a value uniformly within the bin.

**Lessons from implementation.** A first version of the pipeline had three problems that would have
invalidated the results: the random seed was set *after* the split, so cached LLM answers were
attached to different students on every run; answers were matched by their position in the reply
rather than by ID; and many answers were degenerate (all probability on one bin, or the same answer
for all 15 students in a batch). Fixing these was necessary before any analysis. This illustrates a
practical point: the method is simple, but the AI-elicitation step needs careful engineering.

**Quality of the LLM predictions** (evaluated against the hidden true outcomes of the 900 AI students;
"LLM" = mean of the LLM's distribution):

| outcome | mean LLM | mean true | correlation | MSE LLM | MSE of the mean | AI-hours slope LLM | true |
|---|---|---|---|---|---|---|---|
| critical thinking | 71.5 | 61.8 | −0.03 | 317 | 138 | **+0.33** | **−0.30** |
| CGPA | 3.19 | 3.05 | −0.01 | 0.28 | 0.15 | +0.022 | +0.008 |

The LLM is too optimistic, its predictions are essentially uncorrelated with the true values, and it
does worse than always predicting the average. Most importantly for our question, it implies that more
AI use goes with **higher** critical thinking, the opposite of the data. This is a demanding but
realistic test of the method: a confident AI prior that is wrong.

# 6. Experimental design

For both outcomes we fit two models: AI hours only, and AI hours plus all covariates. We use $n = 300$
and $n = 100$ labelled students and a grid of $\alpha \in \{0, 1, 3, 10, 30, 100, 300, m, 1000\}$.
We compare: real data only ($\alpha = 0$), 1:1 pooling ($\alpha = m$, i.e. treating AI labels as
real), and $\alpha$ chosen by CV or by coverage calibration, for both the posterior bootstrap and the
power prior.

To evaluate the methods we repeat the analysis on **100 random splits**. In each split 600 of the 900
AI-labelled students form the AI data, the labelled students are drawn from the remaining real rows,
and the "truth" is the least-squares coefficient on all students *not* used as labelled data. We
report bias, root mean squared error (RMSE), coverage of the 90% interval, and how often the estimate
has the correct sign.

# 7. Results

## 7.1 One split

Table 1 shows the posterior for the AI-hours coefficient on the saved split, and Figure 2 shows how
it changes with $\alpha$.

**Table 1.** Posterior mean [90% interval] of the AI-hours coefficient, critical-thinking score.

| model | n | truth | real data only | 1:1 pooling | power prior, $\alpha$ by CV | posterior bootstrap, $\alpha$ by calibration |
|---|---|---|---|---|---|---|
| AI hours only | 300 | −0.295 | −0.318 [−0.541, −0.094] | **+0.182** [−0.026, 0.391] | −0.318 [−0.541, −0.094] ($\alpha$=0) | −0.258 [−0.479, 0.009] ($\alpha$=30) |
| AI hours only | 100 | −0.305 | −0.265 [−0.607, 0.077] | **+0.290** [0.052, 0.527] | −0.258 [−0.606, 0.091] ($\alpha$=1) | −0.099 [−0.506, 0.317] ($\alpha$=30) |
| all covariates | 300 | −0.381 | −0.687 [−1.14, −0.23] | −0.093 [−0.53, 0.345] | −0.659 [−1.14, −0.175] ($\alpha$=10) | −0.479 [−1.04, 0.142] ($\alpha$=100) |

![**Figure 2.** Posterior mean and 90% interval of the AI-hours coefficient as a function of $\alpha$
(the effective number of AI data points), 300 labelled students, saved split. Blue: power prior;
orange: posterior bootstrap (Dirichlet-process prior). Dashed line: truth. Stars and triangles mark the
$\alpha$ chosen by cross-validation and by coverage calibration.](../regression/output/beta_ai_vs_alpha_n300.png){width=100%}

With 1:1 pooling the 900 AI rows outweigh the real data and the estimated effect becomes **positive**;
with 100 labelled students the interval even excludes zero on the wrong side. CV gives the AI rows
almost no weight and stays close to the real-data estimate. Coverage calibration chooses a larger
$\alpha$ and moves the estimate towards the LLM.

## 7.2 100 random splits

**Table 2.** Performance over 100 random splits, AI-hours coefficient (90% intervals).

| outcome, model | n | method | mean $\alpha$ | bias | RMSE | coverage | correct sign |
|---|---|---|---|---|---|---|---|
| score, AI hours only | 300 | real data only | 0 | 0.012 | 0.146 | 0.91 | 0.99 |
| | | 1:1 pooling | 600 | 0.432 | 0.442 | **0.01** | **0.02** |
| | | power prior, CV | 1.3 | 0.015 | 0.146 | 0.91 | 0.99 |
| | | posterior bootstrap, calibration | 53 | 0.095 | 0.191 | 0.86 | 0.95 |
| score, AI hours only | 100 | real data only | 0 | 0.052 | 0.293 | 0.85 | 0.82 |
| | | 1:1 pooling | 600 | 0.557 | 0.563 | **0.00** | **0.00** |
| | | power prior, CV | 1.3 | 0.060 | 0.292 | 0.86 | 0.80 |
| | | posterior bootstrap, calibration | 180 | 0.231 | 0.377 | **0.73** | 0.58 |
| score, all covariates | 100 | real data only | 0 | −0.049 | 0.632 | 0.84 | 0.78 |
| | | power prior, CV | 14.8 | 0.043 | 0.511 | 0.94 | 0.81 |
| CGPA, AI hours only | 300 | real data only | 0 | 0.000 | 0.0050 | 0.87 | 0.97 |
| | | 1:1 pooling | 600 | 0.009 | 0.0097 | 0.48 | 1.00 |
| | | power prior, CV | 1.0 | 0.000 | 0.0050 | 0.88 | 0.98 |
| CGPA, all covariates | 300 | real data only | 0 | −0.002 | 0.0114 | 0.81 | 0.76 |
| | | 1:1 pooling | 600 | −0.004 | 0.0071 | 1.00 | 0.91 |
| | | posterior bootstrap, calibration | 958 | −0.004 | 0.0072 | 1.00 | 0.90 |

The complete table is `replicates_summary.csv`. The main findings are:

1. **Treating AI labels as real data fails.** For critical thinking, 1:1 pooling gives the wrong sign
   in 98–100% of splits and its 90% interval almost never covers the truth.
2. **Choosing $\alpha$ by cross-validation protects the analysis.** CV gives the AI rows a weight
   of about one data point in the AI-hours-only models, so the results match the real-data analysis.
   In the model with many covariates and only 100 labelled students, a small AI weight
   ($\alpha \approx 15$) even improves the results (RMSE 0.63 → 0.51, coverage 0.84 → 0.94),
   because the AI rows stabilise the estimates of the ~35 other coefficients.
3. **Coverage calibration is too permissive when the AI is biased and labels are few.** Its bootstrap
   estimate of coverage tracks the actual coverage well on average (Figure 3, compare the paper's
   Figure 3), but the rule "choose the largest acceptable $\alpha$" is noisy and
   often overshoots. With 100 labelled students it lets the LLM's bias through: coverage 0.73 and the
   correct sign in only 58% of splits. The reason is that inside the bootstrap, the "truth" is the
   noisy estimate from the small labelled sample, so a moderate bias is hidden by the wide intervals.
4. **The posterior bootstrap and the power prior give almost the same estimates.** With large
   $\alpha$ the posterior bootstrap gives wider, more honest intervals (e.g. coverage 0.11 vs 0.01 at
   1:1 pooling), in line with the paper's argument for the non-parametric approach.
5. **The AI prior can help where the LLM happens to be right.** For CGPA with all covariates the
   LLM's adjusted effect is close to zero, as is the true one, and a large AI weight reduces the RMSE
   from 0.011 to 0.007 with full coverage (Figure 4, bottom right).

![**Figure 3.** Coverage of the 90% intervals over 100 random splits (300 labelled students) as a
function of $\alpha$. Solid: actual coverage (using the truth); dashed: bootstrap estimate used by the
calibration rule (labelled data only). Dotted line: nominal 90%.](../regression/output/calibration_n300.png){width=100%}

![**Figure 4.** Root mean squared error of the AI-hours coefficient over 100 random splits as a function
of $\alpha$, for the power prior (blue) and the posterior bootstrap (orange), with 300 (solid) and 100
(dashed) labelled students.](../regression/output/replicates_rmse_by_alpha.png){width=100%}

# 8. Sensitivity to the LLM and the prompt

**[TODO – fill in when v3 has finished.]** To check whether the results depend on our choice of LLM and
prompt, we repeated the elicitation in the style of the paper (Section 5.1.2 and Appendix A.1): one
student per call, a detailed prompt that explains the meaning and scale of every feature, temperature
0.7, and a larger reasoning model (`gpt-oss-120b`, medium reasoning effort). Unlike the paper, we did
not add expert guidance on how each feature affects the outcomes, because this would mean writing our
own answer into the prior. Results: **[TODO: LLM quality table v2 vs v3, and key rows of the
100-split table with v3 labels — generated automatically in Section 5 of `regression_analysis.html`].**

# 9. Application to a multiclass outcome

**[TODO – classification part.]** Suggested structure, mirroring Sections 4–8: the outcome and its
classes, how the LLM probabilities were elicited, the model and loss (e.g. multinomial logistic
regression with cross-entropy loss in the posterior bootstrap), how $\alpha$ was chosen, results, and
the main findings. Where possible use the same split, seed and $\alpha$ grid as the regression part, so
the two parts can be compared in the discussion.

# 10. Discussion

**Advantages.** The method is simple to implement (a weighted least-squares fit repeated $B$ times),
does not require a correct parametric model, gives fully Bayesian uncertainty quantification, and has a
single interpretable tuning parameter: $\alpha$, the number of data points the AI is worth. It also
uses the AI's full predictive distribution rather than a single predicted value.

**Limitations and failure modes observed in our application.**

* *Bias is not corrected, only weighted.* When the AI is biased, any $\alpha > 0$ moves the posterior
  towards the wrong answer (Theorem 1). Our LLM reversed the sign of the effect, so the safe choice
  was $\alpha \approx 0$.
* *The choice of $\alpha$ is critical.* Fixing $\alpha$ in advance, in particular $\alpha = m$, gave
  confidently wrong conclusions. Data-driven choices work only if the labelled sample can detect the
  bias: CV worked well here, while coverage calibration was too permissive with $n = 100$.
* *No gain without AI knowledge.* On this (probably simulated) dataset the LLM had no useful
  knowledge, so the method could at best protect the analysis. The potential benefit shown in the
  paper requires an AI model that is better than chance at the task.
* *Dependence on elicitation.* The prior depends on the LLM, the prompt and the output format; our
  first implementation shows how easily the elicitation step can go wrong.

**Practical recommendations.** Never treat AI labels as real data; evaluate the AI's predictions on
the labelled data first; choose $\alpha$ out-of-sample and report results for a range of $\alpha$;
and be cautious with coverage calibration when the labelled sample is small.

# 11. Conclusion

We applied the AI-powered Bayesian inference framework to estimate the effect of AI use on students'
critical thinking and CGPA, using an LLM to build the prior. The LLM's predictions were overconfident
and implied the wrong direction of the effect. Treating them as real data reversed the conclusion,
while the Dirichlet-process AI prior with $\alpha$ chosen by cross-validation correctly gave them
almost no weight and kept valid inference. The application illustrates both the main strength of the
approach, an explicit and tunable trust in the AI, and its main limitation, that this trust must be
learned from the labelled data.

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
* **[TODO: dataset reference]**
