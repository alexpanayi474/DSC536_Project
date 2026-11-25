# ==========================================================================
# finalize_v3.R -- turn the v3 LLM answers completed so far into the analysis files.
# No API calls. Run it AFTER the last run of llm_prior_labels_v3.R (that script re-saves
# llm_split_v3.rds with all 900 students each time it starts).
#
# The v3 run (one student per call, gpt-oss-120b) is limited by Groq's free daily token
# quota, so it labels only part of the 900 AI students. Here the AI data are restricted to
# the students with a valid answer. The paper treats the number of AI rows m as a free choice
# (a truncation level, not the prior strength, which is alpha), so this is allowed.
#
# Writes (data/llm/): llm_split_v3.rds  (ai_ids = completed students only)
#                     llm_probs_v3.rds, data_ai_complete_v3.rds
# Then:  Sys.setenv(LLM_TAG = "v3"); source("regression/regression_power_prior.R")
# ==========================================================================

library(jsonlite)
DATA_DIR <- Sys.getenv("CAMPUS_DATA_DIR", "data")
LLM_DIR  <- Sys.getenv("LLM_DIR", file.path(DATA_DIR, "llm"))
TAG      <- "v3"
SEED     <- 30175
OUTCOMES <- c("cgpa", "critical_thinking_score")
DROP_COLS <- c("uses_ai", "paid_ai_subscription", "prompt_skill_level", "time_saved_pct",
               "ai_dependency_score", "ai_dependency_level", "integrity_concern_level",
               "flagged_by_ai_detector", "would_recommend_ai")
P_GPA <- paste0("p_cgpa_", 1:8); P_SCORE <- paste0("p_score_", 1:10)

data <- read.csv(file.path(DATA_DIR, "campus_ai_pulse_2026.csv"), stringsAsFactors = TRUE)
data <- data[, setdiff(names(data), DROP_COLS)]
data$student_id <- as.character(data$student_id)

split_v2 <- readRDS(file.path(LLM_DIR, "llm_split_v2.rds"))
cache    <- readRDS(file.path(LLM_DIR, "llm_responses_v3.rds"))

# same parser as llm_prior_labels_v3.R
parse_one <- function(txt) {
  if (is.null(txt) || is.na(txt)) return(NULL)
  json <- regmatches(txt, regexpr("\\{[\\s\\S]*\\}", txt, perl = TRUE))
  js <- if (length(json)) tryCatch(fromJSON(json, simplifyVector = FALSE), error = function(e) NULL)
  g  <- suppressWarnings(as.numeric(unlist(js$cgpa)))
  sc <- suppressWarnings(as.numeric(unlist(js$score)))
  if (length(g) != 8 || length(sc) != 10 || anyNA(c(g, sc)) || any(c(g, sc) < 0) ||
      sum(g) <= 0 || sum(sc) <= 0) return(NULL)
  c(sum_cgpa_raw = sum(g), sum_score_raw = sum(sc),
    setNames(g / sum(g), P_GPA), setNames(sc / sum(sc), P_SCORE))
}

# completed students, kept in the order of the v2 split
rows <- lapply(split_v2$ai_ids, function(id) parse_one(cache[[id]]$text))
done <- !vapply(rows, is.null, TRUE)
ai_ids <- split_v2$ai_ids[done]
cat("v3 students with a valid answer:", length(ai_ids), "of", length(split_v2$ai_ids), "\n")
if (length(ai_ids) < 300) stop("Fewer than 300 completed students: keep running llm_prior_labels_v3.R first.")

split_v3 <- split_v2
split_v3$ai_ids <- ai_ids
split_v3$ai_ids_unlabelled <- split_v2$ai_ids[!done]   # not used as AI rows; they become test rows
split_v3$note <- "v3: AI rows restricted to students completed within the Groq free daily quota"
saveRDS(split_v3, file.path(LLM_DIR, paste0("llm_split_", TAG, ".rds")))

probs <- data.frame(student_id = ai_ids, do.call(rbind, rows[done]))
probs$exp_cgpa  <- as.vector(as.matrix(probs[P_GPA])   %*% seq(0.25, 3.75, 0.5))
probs$exp_score <- as.vector(as.matrix(probs[P_SCORE]) %*% seq(5, 95, 10))
saveRDS(probs, file.path(LLM_DIR, paste0("llm_probs_", TAG, ".rds")))

set.seed(SEED)
draw <- function(p, lo, hi) { k <- sample(length(p), 1, prob = p); runif(1, lo[k], hi[k]) }
ai <- data[match(ai_ids, data$student_id), setdiff(names(data), OUTCOMES)]
ai$cgpa <- round(apply(probs[P_GPA], 1, draw, seq(0, 3.5, .5), seq(.5, 4, .5)), 2)
ai$critical_thinking_score <- round(apply(probs[P_SCORE], 1, draw, seq(0, 90, 10), seq(10, 100, 10)), 1)
saveRDS(ai, file.path(LLM_DIR, paste0("data_ai_complete_", TAG, ".rds")))

# quick quality check against v2 on the same students (true labels used only for evaluation)
v2 <- readRDS(file.path(LLM_DIR, "llm_probs_v2.rds")); v2 <- v2[match(ai_ids, v2$student_id), ]
tr <- data[match(ai_ids, data$student_id), ]
q <- function(es, eg) c(mean_score = mean(es), cor_score = cor(es, tr$critical_thinking_score),
                        slope_score = unname(coef(lm(es ~ tr$ai_hours_per_week))[2]),
                        mean_cgpa = mean(eg), cor_cgpa = cor(eg, tr$cgpa),
                        slope_cgpa = unname(coef(lm(eg ~ tr$ai_hours_per_week))[2]))
print(round(rbind(v2 = q(v2$exp_score, v2$exp_cgpa), v3 = q(probs$exp_score, probs$exp_cgpa),
                  truth = q(tr$critical_thinking_score, tr$cgpa)), 3))
cat("\nSaved llm_split_v3.rds, llm_probs_v3.rds, data_ai_complete_v3.rds in", normalizePath(LLM_DIR), "\n")
cat("Next: Sys.setenv(LLM_TAG = \"v3\"); source(\"regression/regression_power_prior.R\")\n")
