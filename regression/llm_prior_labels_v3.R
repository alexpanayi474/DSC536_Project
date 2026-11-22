# ==========================================================================
# llm_prior_labels_v3.R -- second LLM labelling run, in the style of the paper
# (O'Hagan & Rockova 2025, Sec. 5.1.2 and Appendix A.1):
#   * ONE student per API call (v2 used batches of 15)
#   * detailed prompt that explains every feature and both outcome scales
#   * temperature 0.7 and medium reasoning effort (as with the paper's o4-mini; v2: 0.4, low)
#   * larger model by default: openai/gpt-oss-120b (v2: gpt-oss-20b)
#   * the full probability distribution per student is kept (paper: AI base measure F_AI)
# Labels the SAME 900 AI students as v2 (read from data/llm/llm_split_v2.rds),
# so v2 and v3 can be compared directly. v2 files are never modified.
#
# Setup: GROQ_API_KEY in ~/.Renviron (already done), httr installed (already done).
# Run (from the project root, in RStudio):  source("regression/llm_prior_labels_v3.R")
#   - It saves progress every 10 students. If it stops (daily token limit, internet,
#     closing RStudio), just run the same line again later: it resumes where it stopped.
#   - When finished, analyse with:
#       Sys.setenv(LLM_TAG = "v3"); source("regression/regression_power_prior.R")
# Test without API calls:  Sys.setenv(LLM_DRY_RUN = "1")
# Other model:             Sys.setenv(LLM_MODEL = "...")   (default openai/gpt-oss-120b)
# ==========================================================================

library(jsonlite)

DATA_DIR    <- Sys.getenv("CAMPUS_DATA_DIR", "data")
LLM_DIR     <- Sys.getenv("LLM_DIR", file.path(DATA_DIR, "llm"))
OUT_DIR     <- Sys.getenv("LLM_OUT_DIR", LLM_DIR)
DRY_RUN     <- identical(Sys.getenv("LLM_DRY_RUN"), "1")
OUT_TAG     <- if (DRY_RUN) "v3dry" else "v3"
SPLIT_TAG   <- "v2"                                  # reuse the v2 split: same 900 students

SEED        <- 30175
# The paper used OpenAI o4-mini (a reasoning model, default reasoning effort "medium") at
# temperature 0.7. o4-mini is not on Groq; gpt-oss-120b is OpenAI's open-weight reasoning model.
MODEL_ID    <- Sys.getenv("LLM_MODEL", "openai/gpt-oss-120b")
TEMPERATURE <- 0.7
REASONING   <- Sys.getenv("LLM_REASONING", "medium")   # "low" uses fewer tokens if the daily limit bites
MAX_TRIES   <- 3          # per student, if the answer is not valid JSON with 8 + 10 numbers
SLEEP_SEC   <- 2.5        # pause between calls (free-tier requests-per-minute limit)
SAVE_EVERY  <- 10
ENDPOINT    <- "https://api.groq.com/openai/v1/chat/completions"
MODELS_URL  <- "https://api.groq.com/openai/v1/models"

DROP_COLS <- c("uses_ai", "paid_ai_subscription", "prompt_skill_level", "time_saved_pct",
               "ai_dependency_score", "ai_dependency_level", "integrity_concern_level",
               "flagged_by_ai_detector", "would_recommend_ai")      # same features as v2
OUTCOMES  <- c("cgpa", "critical_thinking_score")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
out_file <- function(name) file.path(OUT_DIR, paste0(name, "_", OUT_TAG, ".rds"))

# --------------------------------------------------------------------------
# 1. Data and the (unchanged) v2 split
# --------------------------------------------------------------------------
data <- read.csv(file.path(DATA_DIR, "campus_ai_pulse_2026.csv"), stringsAsFactors = TRUE)
data <- data[, setdiff(names(data), DROP_COLS)]
data$student_id <- as.character(data$student_id)

split <- readRDS(file.path(LLM_DIR, paste0("llm_split_", SPLIT_TAG, ".rds")))
saveRDS(split, out_file("llm_split"))                # same split, saved under the v3 tag
ai_ids  <- split$ai_ids
data_ai <- data[match(ai_ids, data$student_id), setdiff(names(data), OUTCOMES)]
feature_cols <- setdiff(names(data_ai), "student_id")

# --------------------------------------------------------------------------
# 2. Detailed prompt (paper style: explain the inputs, one case per prompt)
# --------------------------------------------------------------------------
FEATURE_DESC <- c(
  age                     = "age in years",
  gender                  = "gender",
  country                 = "country where the student studies",
  university_type         = "Public or Private university",
  program                 = "field of study",
  year_of_study           = "year of study (1 = first year, ..., 4 = fourth year)",
  uni_ai_policy           = "the university's policy on using AI tools in coursework",
  primary_ai_tool         = "the AI tool the student uses most ('No AI tool' = the student does not use AI)",
  primary_use_case        = "what the student mainly uses AI for ('No AI use' = the student does not use AI)",
  ai_hours_per_week       = "hours per week the student spends using AI tools",
  ai_assignment_share_pct = "percentage (0-100) of the student's assignment work done with AI help",
  study_hours_per_week    = "hours per week of independent study")
stopifnot(all(feature_cols %in% names(FEATURE_DESC)))

# Same structure as the paper's prompt (Appendix A.1): role and task -> how to read the input
# -> meaning and scale of every feature -> what to estimate -> output format.
# Deliberate difference: the paper adds "Detailed Disease Profiles" (expert guidance on how each
# feature relates to each class). The analogue here would be telling the LLM how AI use affects
# grades, i.e. writing our own answer into the prior, so the LLM must rely on its own knowledge.
SYSTEM_PROMPT <- paste0(
  "You are an advanced AI educational assessment assistant. Your current task is to analyze a set ",
  "of characteristics of a university student, related to their study habits and their use of ",
  "generative AI tools, and estimate the probability distribution of two academic outcomes for this ",
  "student. Your analysis should be based on a deep understanding of how these characteristics ",
  "typically relate to students' academic performance. The input will be a list of characteristics ",
  "with their values for ONE student. Pay close attention to the meaning and scale of each one and ",
  "judge this student individually.\n\n",
  "Understanding the Input Features:\n",
  paste0(feature_cols, ": ", FEATURE_DESC[feature_cols], ".", collapse = "\n"), "\n\n",
  "Outcomes to Estimate:\n",
  "cgpa: the student's cumulative grade point average on a 0-4 scale. Estimate the probability that ",
  "it falls in each of 8 bins: [0.0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2.0-2.5, 2.5-3.0, 3.0-3.5, 3.5-4.0].\n",
  "score: the student's score on a standardised critical-thinking assessment on a 0-100 scale. ",
  "Estimate the probability that it falls in each of 10 bins: ",
  "[0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100].\n",
  "Each set of probabilities must sum to 1 and should reflect your genuine uncertainty.\n\n",
  "Output Format:\n",
  "Return ONLY a JSON object of the form {\"cgpa\": [8 probabilities], \"score\": [10 probabilities]}, ",
  "with the probabilities in the bin order given above and no other text.")

student_text <- function(i) paste0("Student:\n",
  paste0(feature_cols, ": ", vapply(feature_cols, function(v) as.character(data_ai[[v]][i]), ""),
         collapse = "\n"))

# --------------------------------------------------------------------------
# 3. API
# --------------------------------------------------------------------------
check_model <- function() {
  key <- Sys.getenv("GROQ_API_KEY")
  if (!nzchar(key)) stop("GROQ_API_KEY is not set. Put it in ~/.Renviron and restart R.")
  r <- httr::GET(MODELS_URL, httr::add_headers(Authorization = paste("Bearer", key)))
  if (httr::status_code(r) != 200)
    stop("Could not list Groq models (HTTP ", httr::status_code(r), "): ",
         substr(httr::content(r, "text", encoding = "UTF-8"), 1, 300))
  ids <- vapply(fromJSON(httr::content(r, "text", encoding = "UTF-8"), simplifyVector = FALSE)$data,
                function(m) m$id, "")
  if (!MODEL_ID %in% ids)
    stop("Model '", MODEL_ID, "' is not available on your Groq account. Available models include:\n  ",
         paste(sort(ids), collapse = "\n  "),
         "\nPick one with Sys.setenv(LLM_MODEL = \"...\") and run again.")
  cat("Model", MODEL_ID, "is available.\n")
}

get_response <- function(user_text) {
  if (DRY_RUN) {                                        # fake valid answer, no network
    rd <- function(k) { x <- rgamma(k, 1); round(x / sum(x), 3) }
    return(as.character(toJSON(list(cgpa = rd(8), score = rd(10)), auto_unbox = TRUE)))
  }
  key  <- Sys.getenv("GROQ_API_KEY")
  body <- toJSON(list(model = MODEL_ID,
                      messages = list(list(role = "system", content = SYSTEM_PROMPT),
                                      list(role = "user", content = user_text)),
                      temperature = TEMPERATURE, max_tokens = 4000, reasoning_effort = REASONING,
                      response_format = list(type = "json_object")), auto_unbox = TRUE)
  for (attempt in 1:5) {
    resp <- tryCatch(httr::POST(ENDPOINT, body = body, encode = "raw",
                                httr::add_headers(Authorization = paste("Bearer", key),
                                                  `Content-Type` = "application/json")),
                     error = function(e) { cat("  [connection error]", conditionMessage(e), "\n"); NULL })
    if (is.null(resp)) { Sys.sleep(10); next }
    sc  <- httr::status_code(resp)
    txt <- httr::content(resp, "text", encoding = "UTF-8")
    if (sc == 200) {
      out <- tryCatch(fromJSON(txt, simplifyVector = FALSE)$choices[[1]]$message$content,
                      error = function(e) NULL)
      if (length(out) && nzchar(trimws(out))) return(out)
    } else if (sc %in% c(401, 403, 404)) {
      stop("Groq returned HTTP ", sc, " (bad key or wrong model): ", substr(txt, 1, 300))
    } else if (sc == 429) {
      if (grepl("tokens per day|TPD|requests per day|RPD", txt, ignore.case = TRUE)) return("DAILY_LIMIT")
      cat("  [rate limit] waiting 20s\n"); Sys.sleep(20)
    } else {
      cat("  [HTTP", sc, "]", substr(txt, 1, 150), "\n"); Sys.sleep(5)
    }
  }
  NA_character_
}

P_GPA <- paste0("p_cgpa_", 1:8); P_SCORE <- paste0("p_score_", 1:10)
parse_one <- function(txt) {
  if (is.na(txt)) return(NULL)
  json <- regmatches(txt, regexpr("\\{[\\s\\S]*\\}", txt, perl = TRUE))
  js <- if (length(json)) tryCatch(fromJSON(json, simplifyVector = FALSE), error = function(e) NULL)
  g  <- suppressWarnings(as.numeric(unlist(js$cgpa)))
  sc <- suppressWarnings(as.numeric(unlist(js$score)))
  if (length(g) != 8 || length(sc) != 10 || anyNA(c(g, sc)) || any(c(g, sc) < 0) ||
      sum(g) <= 0 || sum(sc) <= 0) return(NULL)
  c(sum_cgpa_raw = sum(g), sum_score_raw = sum(sc),
    setNames(g / sum(g), P_GPA), setNames(sc / sum(sc), P_SCORE))
}

# --------------------------------------------------------------------------
# 4. Run: one call per student, resuming from the cache
# --------------------------------------------------------------------------
if (!DRY_RUN) check_model()
cat("System prompt sent with every call:\n----\n", SYSTEM_PROMPT, "\n----\n", sep = "")

f_cache <- out_file("llm_responses")
cache   <- if (file.exists(f_cache)) readRDS(f_cache) else list()
todo    <- which(!vapply(ai_ids, function(id) isTRUE(cache[[id]]$ok), TRUE))
cat(length(ai_ids) - length(todo), "of", length(ai_ids), "students already done;",
    length(todo), "to go. Model:", MODEL_ID, "\n")

t0 <- Sys.time(); n_done <- 0
for (i in todo) {
  id <- ai_ids[i]; best <- list(text = NA_character_, ok = FALSE, tries = 0)
  for (try in seq_len(MAX_TRIES)) {
    txt <- get_response(student_text(i))
    if (identical(txt, "DAILY_LIMIT")) {
      saveRDS(cache, f_cache)
      stop(sprintf("Groq daily limit reached after %d students in this session. Progress saved (%d / %d done). ",
                   n_done, sum(vapply(cache, function(x) isTRUE(x$ok), TRUE)), length(ai_ids)),
           "Run the same source() line again tomorrow (or later today) to continue.", call. = FALSE)
    }
    best <- list(text = txt, ok = !is.null(parse_one(txt)), tries = try)
    if (best$ok) break
  }
  cache[[id]] <- best; n_done <- n_done + 1
  if (n_done %% SAVE_EVERY == 0) saveRDS(cache, f_cache)
  if (n_done %% 25 == 0) {
    el <- as.numeric(Sys.time() - t0, units = "mins")
    cat(sprintf("  %d / %d this session (%.0f min, ~%.0f min left if no limit)\n",
                n_done, length(todo), el, el / n_done * (length(todo) - n_done)))
  }
  if (!DRY_RUN) Sys.sleep(SLEEP_SEC)
}
saveRDS(cache, f_cache)

# --------------------------------------------------------------------------
# 5. Probabilities + sampled labels, same format as v2
# --------------------------------------------------------------------------
rows  <- lapply(ai_ids, function(id) parse_one(if (is.null(cache[[id]])) NA_character_ else cache[[id]]$text))
n_bad <- sum(vapply(rows, is.null, TRUE))
if (n_bad > 0) {
  cat("\n", n_bad, "students still have no valid answer. Run the script again to retry only those.\n")
} else {
  probs <- data.frame(student_id = ai_ids, do.call(rbind, rows))
  probs$exp_cgpa  <- as.vector(as.matrix(probs[P_GPA])   %*% seq(0.25, 3.75, 0.5))
  probs$exp_score <- as.vector(as.matrix(probs[P_SCORE]) %*% seq(5, 95, 10))
  saveRDS(probs, out_file("llm_probs"))

  set.seed(SEED)
  draw <- function(p, lo, hi) { k <- sample(length(p), 1, prob = p); runif(1, lo[k], hi[k]) }
  data_ai_complete <- data_ai
  data_ai_complete$cgpa <- round(apply(probs[P_GPA], 1, draw, seq(0, 3.5, .5), seq(.5, 4, .5)), 2)
  data_ai_complete$critical_thinking_score <- round(apply(probs[P_SCORE], 1, draw, seq(0, 90, 10), seq(10, 100, 10)), 1)
  saveRDS(data_ai_complete, out_file("data_ai_complete"))

  # quality report (true labels used only here, for evaluation)
  truth <- data[match(ai_ids, data$student_id), ]
  q <- data.frame(outcome = OUTCOMES,
                  mean_llm = c(mean(probs$exp_cgpa), mean(probs$exp_score)),
                  mean_true = c(mean(truth$cgpa), mean(truth$critical_thinking_score)),
                  sd_llm = c(sd(probs$exp_cgpa), sd(probs$exp_score)),
                  sd_true = c(sd(truth$cgpa), sd(truth$critical_thinking_score)),
                  cor_true = c(cor(probs$exp_cgpa, truth$cgpa), cor(probs$exp_score, truth$critical_thinking_score)),
                  mse_llm = c(mean((probs$exp_cgpa - truth$cgpa)^2), mean((probs$exp_score - truth$critical_thinking_score)^2)),
                  mse_mean = c(var(truth$cgpa), var(truth$critical_thinking_score)))
  f_v2 <- file.path(LLM_DIR, "llm_probs_v2.rds")
  if (file.exists(f_v2)) {
    v2 <- readRDS(f_v2); v2 <- v2[match(ai_ids, v2$student_id), ]
    q$cor_with_v2 <- c(cor(probs$exp_cgpa, v2$exp_cgpa), cor(probs$exp_score, v2$exp_score))
  }
  slope <- function(y) coef(lm(y ~ truth$ai_hours_per_week))[2]
  q$ai_hours_slope_llm  <- c(slope(probs$exp_cgpa), slope(probs$exp_score))
  q$ai_hours_slope_true <- c(slope(truth$cgpa), slope(truth$critical_thinking_score))
  cat("\n===== LLM label quality (", OUT_TAG, ",", MODEL_ID, ") =====\n")
  cat("One-hot answers: cgpa", sum(apply(probs[P_GPA], 1, max) > 0.99),
      "| score", sum(apply(probs[P_SCORE], 1, max) > 0.99), "\n")
  print(q, digits = 3, row.names = FALSE)
  cat("(mse_mean = error of always predicting the true mean; mse_llm should be below it)\n")
  cat("Saved:", paste(basename(c(out_file("llm_split"), f_cache, out_file("llm_probs"),
                                 out_file("data_ai_complete"))), collapse = ", "), "in", normalizePath(OUT_DIR), "\n")
  cat("Next: Sys.setenv(LLM_TAG = \"v3\"); source(\"regression/regression_power_prior.R\")\n")
}
