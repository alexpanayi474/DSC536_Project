# ==========================================================================
# llm_bins_sensitivity.R -- does the way the LLM states its uncertainty matter?
#
# Same prompt, model (gpt-oss-120b), provider, temperature (0.7) and reasoning effort (medium)
# as llm_prior_labels_v3.R; only the output format changes:
#   SENS_SCHEME = "fine"   : CGPA 16 bins of width 0.25, score 20 bins of width 5
#                            (v3: 8 bins of 0.5 and 10 bins of 10; bins chosen from the scales only)
#   SENS_SCHEME = "meansd" : a predictive mean and standard deviation for each outcome
# Run on the same 300 AI students for both schemes (the first 300 of the fixed random split,
# chosen before looking at any result), and compared with the v3 answers for those students.
# This is a sensitivity check only: the main analysis keeps the v3 labels.
#
# Run (project root, RStudio), one scheme at a time:
#   Sys.setenv(LLM_PROVIDER = "openrouter", SENS_SCHEME = "fine");   source("regression/llm_bins_sensitivity.R")
#   Sys.setenv(LLM_PROVIDER = "openrouter", SENS_SCHEME = "meansd"); source("regression/llm_bins_sensitivity.R")
# It resumes where it stopped. Output: data/llm/llm_responses_sens_<scheme>.rds and
# regression/output_sensitivity/elicitation_comparison.csv (all schemes done so far).
# Test without API calls: Sys.setenv(LLM_DRY_RUN = "1")
# ==========================================================================

library(jsonlite)
DATA_DIR <- Sys.getenv("CAMPUS_DATA_DIR", "data")
LLM_DIR  <- Sys.getenv("LLM_DIR", file.path(DATA_DIR, "llm"))
OUT_DIR  <- Sys.getenv("SENS_OUT_DIR", file.path("regression", "output_sensitivity"))
DRY_RUN  <- identical(Sys.getenv("LLM_DRY_RUN"), "1")
SCHEME   <- Sys.getenv("SENS_SCHEME", "fine")
PROVIDER <- Sys.getenv("LLM_PROVIDER", "openrouter")
stopifnot(SCHEME %in% c("fine", "meansd"), PROVIDER %in% c("groq", "openrouter"))
N_SENS   <- 300
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

MODEL_ID    <- "openai/gpt-oss-120b"
TEMPERATURE <- 0.7
REASONING   <- "medium"
MAX_TRIES   <- 3
API <- switch(PROVIDER,
  groq = list(endpoint = "https://api.groq.com/openai/v1/chat/completions", key_var = "GROQ_API_KEY", sleep = 2.5),
  openrouter = list(endpoint = "https://openrouter.ai/api/v1/chat/completions", key_var = "OPENROUTER_API_KEY",
                    sleep = 0.3, order = "groq"))

DROP_COLS <- c("uses_ai", "paid_ai_subscription", "prompt_skill_level", "time_saved_pct",
               "ai_dependency_score", "ai_dependency_level", "integrity_concern_level",
               "flagged_by_ai_detector", "would_recommend_ai")
OUTCOMES  <- c("cgpa", "critical_thinking_score")

data <- read.csv(file.path(DATA_DIR, "campus_ai_pulse_2026.csv"), stringsAsFactors = TRUE)
data <- data[, setdiff(names(data), DROP_COLS)]
data$student_id <- as.character(data$student_id)
split <- readRDS(file.path(LLM_DIR, "llm_split_v2.rds"))
ids   <- split$ai_ids[seq_len(N_SENS)]
data_target  <- data[match(ids, data$student_id), setdiff(names(data), OUTCOMES)]
feature_cols <- setdiff(names(data_target), "student_id")

# --------------------------------------------------------------------------
# Prompt: identical to v3 except the "Outcomes to Estimate" and "Output Format" parts
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
HEAD <- paste0(
  "You are an advanced AI educational assessment assistant. Your current task is to analyze a set ",
  "of characteristics of a university student, related to their study habits and their use of ",
  "generative AI tools, and estimate the probability distribution of two academic outcomes for this ",
  "student. Your analysis should be based on a deep understanding of how these characteristics ",
  "typically relate to students' academic performance. The input will be a list of characteristics ",
  "with their values for ONE student. Pay close attention to the meaning and scale of each one and ",
  "judge this student individually.\n\n",
  "Understanding the Input Features:\n",
  paste0(feature_cols, ": ", FEATURE_DESC[feature_cols], ".", collapse = "\n"), "\n\n")

bins_text <- function(lo, hi, digits) paste0("[", paste0(formatC(lo, format = "f", digits = digits), "-",
                                                          formatC(hi, format = "f", digits = digits), collapse = ", "), "]")
if (SCHEME == "fine") {
  G_LO <- seq(0, 3.75, 0.25); G_HI <- G_LO + 0.25; S_LO <- seq(0, 95, 5); S_HI <- S_LO + 5
  TAIL <- paste0(
    "Outcomes to Estimate:\n",
    "cgpa: the student's cumulative grade point average on a 0-4 scale. Estimate the probability that ",
    "it falls in each of 16 bins: ", bins_text(G_LO, G_HI, 2), ".\n",
    "score: the student's score on a standardised critical-thinking assessment on a 0-100 scale. ",
    "Estimate the probability that it falls in each of 20 bins: ", bins_text(S_LO, S_HI, 0), ".\n",
    "Each set of probabilities must sum to 1 and should reflect your genuine uncertainty.\n\n",
    "Output Format:\n",
    "Return ONLY a JSON object of the form {\"cgpa\": [16 probabilities], \"score\": [20 probabilities]}, ",
    "with the probabilities in the bin order given above and no other text.")
} else {
  TAIL <- paste0(
    "Outcomes to Estimate:\n",
    "cgpa: the student's cumulative grade point average on a 0-4 scale. Give the mean and the standard ",
    "deviation of your predictive distribution for it.\n",
    "score: the student's score on a standardised critical-thinking assessment on a 0-100 scale. Give the ",
    "mean and the standard deviation of your predictive distribution for it.\n",
    "The standard deviations should reflect your genuine uncertainty.\n\n",
    "Output Format:\n",
    "Return ONLY a JSON object of the form {\"cgpa\": {\"mean\": number, \"sd\": number}, ",
    "\"score\": {\"mean\": number, \"sd\": number}} and no other text.")
}
SYSTEM_PROMPT <- paste0(HEAD, TAIL)
student_text <- function(i) paste0("Student:\n",
  paste0(feature_cols, ": ", vapply(feature_cols, function(v) as.character(data_target[[v]][i]), ""), collapse = "\n"))

# --------------------------------------------------------------------------
# API call and parsing
# --------------------------------------------------------------------------
get_response <- function(user_text) {
  if (DRY_RUN) {
    if (SCHEME == "fine") { rd <- function(k) { x <- rgamma(k, 1); round(x / sum(x), 3) }
      return(as.character(toJSON(list(cgpa = rd(16), score = rd(20)), auto_unbox = TRUE))) }
    return(as.character(toJSON(list(cgpa = list(mean = runif(1, 2.5, 3.5), sd = 0.4),
                                    score = list(mean = runif(1, 50, 70), sd = 12)), auto_unbox = TRUE)))
  }
  key <- Sys.getenv(API$key_var)
  if (!nzchar(key)) stop(API$key_var, " is not set. Put it in ~/.Renviron and restart R (or readRenviron('~/.Renviron')).")
  req <- list(model = MODEL_ID, messages = list(list(role = "system", content = SYSTEM_PROMPT),
                                                list(role = "user", content = user_text)),
              temperature = TEMPERATURE, max_tokens = 4000, response_format = list(type = "json_object"))
  if (PROVIDER == "groq") req$reasoning_effort <- REASONING else {
    req$reasoning <- list(effort = REASONING); req$provider <- list(order = as.list(API$order), allow_fallbacks = FALSE) }
  body <- toJSON(req, auto_unbox = TRUE)
  for (attempt in 1:5) {
    resp <- tryCatch(httr::POST(API$endpoint, body = body, encode = "raw",
                                httr::add_headers(Authorization = paste("Bearer", key), `Content-Type` = "application/json")),
                     error = function(e) { cat("  [connection error]", conditionMessage(e), "\n"); NULL })
    if (is.null(resp)) { Sys.sleep(10); next }
    sc <- httr::status_code(resp); txt <- httr::content(resp, "text", encoding = "UTF-8")
    if (sc == 200) {
      out <- tryCatch(fromJSON(txt, simplifyVector = FALSE)$choices[[1]]$message$content, error = function(e) NULL)
      if (length(out) && nzchar(trimws(out))) return(out)
      Sys.sleep(3)
    } else if (sc == 402) { stop("Not enough credit (HTTP 402).")
    } else if (sc %in% c(401, 403, 404)) { stop("HTTP ", sc, ": ", substr(txt, 1, 300))
    } else if (sc == 429) { if (PROVIDER == "groq" && grepl("per day", txt, ignore.case = TRUE)) return("DAILY_LIMIT")
      Sys.sleep(20)
    } else { cat("  [HTTP", sc, "]", substr(txt, 1, 150), "\n"); Sys.sleep(5) }
  }
  NA_character_
}

# returns c(mean_cgpa, sd_cgpa, mean_score, sd_score) or NULL
parse_one <- function(txt) {
  if (is.null(txt) || is.na(txt)) return(NULL)
  json <- regmatches(txt, regexpr("\\{[\\s\\S]*\\}", txt, perl = TRUE))
  js <- if (length(json)) tryCatch(fromJSON(json, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(js)) return(NULL)
  if (SCHEME == "fine") {
    g <- suppressWarnings(as.numeric(unlist(js$cgpa))); s <- suppressWarnings(as.numeric(unlist(js$score)))
    if (length(g) != 16 || length(s) != 20 || anyNA(c(g, s)) || any(c(g, s) < 0) || sum(g) <= 0 || sum(s) <= 0) return(NULL)
    mom <- function(p, lo, hi) { p <- p / sum(p); m1 <- sum(p * (lo + hi) / 2)
      c(m1, sqrt(max(sum(p * (lo^2 + lo * hi + hi^2) / 3) - m1^2, 0))) }
    return(c(mom(g, G_LO, G_HI), mom(s, S_LO, S_HI)))
  }
  v <- suppressWarnings(as.numeric(c(js$cgpa$mean, js$cgpa$sd, js$score$mean, js$score$sd)))
  if (length(v) != 4 || anyNA(v) || v[2] <= 0 || v[4] <= 0 || v[1] < 0 || v[1] > 4 || v[3] < 0 || v[3] > 100) return(NULL)
  v
}

# --------------------------------------------------------------------------
# Run (resumes from the cache)
# --------------------------------------------------------------------------
tag     <- paste0("sens_", SCHEME, if (DRY_RUN) "_dry" else "")
f_cache <- file.path(LLM_DIR, paste0("llm_responses_", tag, ".rds"))
cache   <- if (file.exists(f_cache)) readRDS(f_cache) else list()
todo    <- which(!vapply(ids, function(id) isTRUE(cache[[id]]$ok), TRUE))
cat("Scheme:", SCHEME, "| provider:", PROVIDER, "|", N_SENS - length(todo), "of", N_SENS, "done,", length(todo), "to go\n")
cat("Prompt (outcome and output part):\n----\n", TAIL, "\n----\n", sep = "")
n_done <- 0
for (i in todo) {
  id <- ids[i]; best <- list(text = NA_character_, ok = FALSE)
  for (try in seq_len(MAX_TRIES)) {
    txt <- get_response(student_text(i))
    if (identical(txt, "DAILY_LIMIT")) { saveRDS(cache, f_cache); stop("Groq daily limit reached; progress saved.") }
    best <- list(text = txt, ok = !is.null(parse_one(txt)), tries = try,
                 provider = if (PROVIDER == "openrouter") "openrouter/groq" else "groq", model = MODEL_ID)
    if (best$ok) break
  }
  cache[[id]] <- best; n_done <- n_done + 1
  if (n_done %% 10 == 0) saveRDS(cache, f_cache)
  if (n_done %% 50 == 0) cat("  ", n_done, "/", length(todo), "\n")
  if (!DRY_RUN) Sys.sleep(API$sleep)
}
saveRDS(cache, f_cache)

# --------------------------------------------------------------------------
# Compare all schemes available so far with v3 on the same students (truth used only here)
# --------------------------------------------------------------------------
truth <- data[match(ids, data$student_id), ]
d_all <- read.csv(file.path(DATA_DIR, "campus_ai_pulse_2026.csv"))
tru   <- d_all[match(ids, d_all$student_id), ]
p3 <- readRDS(file.path(LLM_DIR, "llm_probs_v3.rds")); p3 <- p3[match(ids, p3$student_id), ]
v3sd <- function(cols, lo, hi) { P <- as.matrix(p3[cols]); m1 <- drop(P %*% ((lo + hi) / 2))
  sqrt(pmax(drop(P %*% ((lo^2 + lo * hi + hi^2) / 3)) - m1^2, 0)) }
runs <- list(v3_coarse_bins = cbind(p3$exp_cgpa, v3sd(paste0("p_cgpa_", 1:8), seq(0, 3.5, .5), seq(.5, 4, .5)),
                                    p3$exp_score, v3sd(paste0("p_score_", 1:10), seq(0, 90, 10), seq(10, 100, 10))))
for (sch in c("fine", "meansd")) {
  f <- file.path(LLM_DIR, paste0("llm_responses_sens_", sch, if (DRY_RUN) "_dry" else "", ".rds"))
  if (!file.exists(f)) next
  cc <- readRDS(f); old <- SCHEME; SCHEME <- sch
  if (sch == "fine") { G_LO <- seq(0, 3.75, 0.25); G_HI <- G_LO + 0.25; S_LO <- seq(0, 95, 5); S_HI <- S_LO + 5 }
  m <- t(vapply(ids, function(id) { r <- parse_one(cc[[id]]$text); if (is.null(r)) rep(NA_real_, 4) else r }, numeric(4)))
  SCHEME <- old
  runs[[if (sch == "fine") "fine_bins" else "mean_sd"]] <- m
}
row <- function(name, m, k, y) {
  ok <- !is.na(m[, k])
  data.frame(format = name, outcome = y, n_valid = sum(ok),
             mean_pred = mean(m[ok, k]), mean_true = mean(tru[[y]][ok]),
             cor_truth = cor(m[ok, k], tru[[y]][ok]), mse = mean((m[ok, k] - tru[[y]][ok])^2),
             stated_sd = mean(m[ok, k + 1]), true_sd = sd(tru[[y]][ok]),
             slope_llm = unname(coef(lm(m[ok, k] ~ tru$ai_hours_per_week[ok]))[2]),
             slope_true = unname(coef(lm(tru[[y]][ok] ~ tru$ai_hours_per_week[ok]))[2]))
}
cmp <- do.call(rbind, c(lapply(names(runs), function(nm) row(nm, runs[[nm]], 3, "critical_thinking_score")),
                        lapply(names(runs), function(nm) row(nm, runs[[nm]], 1, "cgpa"))))
if (!DRY_RUN) write.csv(cmp, file.path(OUT_DIR, "elicitation_comparison.csv"), row.names = FALSE)
old <- options(width = 200)
cat("\n===== Elicitation format comparison,", N_SENS, "AI students (same students for every format) =====\n")
print(cmp, digits = 3, row.names = FALSE)
cat("stated_sd = average SD the LLM states for one student (its uncertainty); true_sd = spread of the real values\n")
options(old)
