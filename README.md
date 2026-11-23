# DSC536 Project – AI-Powered Bayesian Inference

Group project on O'Hagan & Ročková (2025), *AI-Powered Bayesian Inference*
([arXiv:2502.19231](https://arxiv.org/abs/2502.19231)): using LLM predictions to build a Bayesian prior.

Dataset: `campus_ai_pulse_2026.csv` (1200 students; AI use, study habits, CGPA, critical-thinking score).
It is a **synthetic** dataset by Muhammad Aamir,
[Campus AI Pulse 2026 on Kaggle](https://www.kaggle.com/datasets/aamir28/student-ai-usage-and-academic-performance-2026),
licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); it is redistributed
here under the same licence, for non-commercial coursework.

## Folder structure

```
data/
  campus_ai_pulse_2026.csv      raw dataset
  llm/                          LLM (Groq, gpt-oss-20b) outputs
    llm_split_v2.rds            known / AI split (student_ids, seed 30175)
    llm_responses_v2.rds        raw LLM answers
    llm_probs_v2.rds            LLM probability distributions per student
    data_ai_complete_v2.rds     AI rows with LLM-sampled labels
    data_known_v2.rds           known (labelled) rows
    risposte_groq_oss20b.rds    old responses from bapr2.R (not used: split not reproducible)
regression/                     linear-regression half (cgpa, critical_thinking_score ~ ai_hours_per_week)
  data_check.R                  dataset quality / suitability checks
  llm_prior_labels.R            asks the LLM for label distributions (needs GROQ_API_KEY)
  regression_power_prior.R      AI-prior analysis: power prior + posterior bootstrap (paper Alg. 1),
                                alpha chosen by CV and by bootstrap coverage calibration
  output/                       tables and figures
classification/                 multiclass half
report/                         final report and slides
bapr2.R                         first draft of the regression pipeline (kept for reference)
```

## How to run (regression)

Open `DSC536_Project.Rproj` (activates renv), then from the project root:

```r
source("regression/data_check.R")              # ~10 s, writes regression/output/data_check/
source("regression/regression_power_prior.R")  # ~20 min, offline, writes regression/output/
```

`regression/llm_prior_labels.R` calls the Groq API and is only needed to regenerate `data/llm/`.
It needs `GROQ_API_KEY` in `~/.Renviron` (never in the code) and `renv::install("httr")`.
