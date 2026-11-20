# ==========================================================================
# llm_prior_labels.R -- regenerate the LLM-imputed labels for the AI rows.
#
# Same idea as sections 2 and 4-7 of bapr2.R (bapr2.R itself is NOT modified), with fixes:
#   * set.seed() BEFORE the shuffle; the split (student_ids) is saved and checked
#   * API key read from the environment (GROQ_API_KEY), never written in code
#   * students sent with their real student_id and matched back BY ID, not position
#   * JSON mode + larger max_tokens (gpt-oss reasoning tokens count toward it)
#   * 401/403/404 stop the script instead of silently producing NA
#   * batches with missing students or 15 identical answers are retried
#   * full per-student probabilities are saved, not just one sampled value
#   * new file names (*_v2.rds): the old risposte_groq_oss20b.rds is never touched
#
# Setup: add GROQ_API_KEY=... to ~/.Renviron (not in the repo), restart R,
#        renv::install("httr") once (bapr2.R needs it too).
# Run:   source("regression/llm_prior_labels.R")  (from the project root) -- resumes from the cache
# Test:  Sys.setenv(LLM_DRY_RUN = "1") -- fake LLM answers, no API calls, tag "dryrun"
# ==========================================================================

library(jsonlite)

DATA_DIR   <- Sys.getenv("CAMPUS_DATA_DIR", "data")                   # where the csv is
OUT_DIR    <- Sys.getenv("LLM_OUT_DIR", file.path(DATA_DIR, "llm"))   # where the LLM outputs go
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
DRY_RUN    <- identical(Sys.getenv("LLM_DRY_RUN"), "1")
OUT_TAG    <- if (DRY_RUN) "dryrun" else "v2"

SEED       <- 30175
N_KNOWN    <- 300
N_AI       <- 900
BATCH_SIZE <- 15
MAX_TRIES  <- 3        # per batch, when the answer is incomplete or degenerate

MODEL_ID   <- "openai/gpt-oss-20b"
ENDPOINT   <- "https://api.groq.com/openai/v1/chat/completions"

# Same columns bapr2.R drops (positions 9, 13, 14, 19-24), by NAME. student_id is kept as the key.
DROP_COLS <- c("uses_ai", "paid_ai_subscription", "prompt_skill_level", "time_saved_pct",
               "ai_dependency_score", "ai_dependency_level", "integrity_concern_level",
               "flagged_by_ai_detector", "would_recommend_ai")
OUTCOMES  <- c("cgpa", "critical_thinking_score")

out_file <- function(name) file.path(OUT_DIR, paste0(name, "_", OUT_TAG, ".rds"))

# --------------------------------------------------------------------------
# 1. Data + reproducible split
# --------------------------------------------------------------------------
data <- read.csv(file.path(DATA_DIR, "campus_ai_pulse_2026.csv"), stringsAsFactors = TRUE)
stopifnot(all(DROP_COLS %in% names(data)), !anyDuplicated(data$student_id),
          N_KNOWN + N_AI <= nrow(data))
data <- data[, setdiff(names(data), DROP_COLS)]
data$student_id <- as.character(data$student_id)

set.seed(SEED)
perm  <- sample(nrow(data))
split <- list(seed = SEED,
              known_ids = data$student_id[perm[seq_len(N_KNOWN)]],
              ai_ids    = data$student_id[perm[N_KNOWN + seq_len(N_AI)]],
              test_ids  = data$student_id[perm[-seq_len(N_KNOWN + N_AI)]])

f_split <- out_file("llm_split")
if (file.exists(f_split)) {
  old <- readRDS(f_split)
  if (!identical(old[c("known_ids", "ai_ids")], split[c("known_ids", "ai_ids")]))
    stop("Split differs from the saved one in ", f_split,
         " (different seed/R version/N_KNOWN?). Cached LLM answers would not match. ",
         "Use a new OUT_TAG or delete the cache deliberately.")
} else saveRDS(split, f_split)

data_known  <- data[match(split$known_ids, data$student_id), ]
data_ai_raw <- data[match(split$ai_ids, data$student_id), setdiff(names(data), OUTCOMES)]
feature_cols <- setdiff(names(data_ai_raw), "student_id")

# --------------------------------------------------------------------------
# 2. Prompt
# --------------------------------------------------------------------------
GPA_LO   <- seq(0, 3.5, 0.5); GPA_HI   <- GPA_LO + 0.5
SCORE_LO <- seq(0, 90, 10);   SCORE_HI <- SCORE_LO + 10

create_batch_prompt <- function(d) {
  rows <- vapply(seq_len(nrow(d)), function(i)
    paste(c(d$student_id[i], vapply(feature_cols, function(v) as.character(d[[v]][i]), "")),
          collapse = "|"), "")
  paste0(
    "You are an educational data analyst. For each university student below, give your predictive ",
    "probability distribution for two outcomes, based only on that student's own features.\n",
    "1) \"cgpa\" (0-4 scale): 8 probabilities for bins ",
    "[0.0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2.0-2.5, 2.5-3.0, 3.0-3.5, 3.5-4.0]\n",
    "2) \"score\" (critical thinking score, 0-100): 10 probabilities for bins ",
    "[0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100]\n",
    "Each list must sum to 1. Reflect your real uncertainty and judge each student individually.\n",
    "Columns: id|", paste(feature_cols, collapse = "|"), "\n",
    "Return only a JSON object {\"students\": [{\"id\": \"<id>\", \"cgpa\": [8 numbers], ",
    "\"score\": [10 numbers]}, ...]} with exactly one entry per student, using the ids given.\n\n",
    paste(rows, collapse = "\n"))
}

# --------------------------------------------------------------------------
# 3. API call (or a fake one for DRY_RUN)
# --------------------------------------------------------------------------
get_response <- function(prompt_text) {
  if (DRY_RUN) {                                   # random but valid answers, no network
    ids <- sub("\\|.*$", "", grep("^S[0-9]+\\|", strsplit(prompt_text, "\n")[[1]], value = TRUE))
    rd  <- function(k) { x <- rgamma(k, 1); round(x / sum(x), 3) }
    return(as.character(toJSON(list(students = lapply(ids, function(id)
      list(id = id, cgpa = rd(8), score = rd(10)))), auto_unbox = TRUE)))
  }
  key <- Sys.getenv("GROQ_API_KEY")
  if (!nzchar(key)) stop("GROQ_API_KEY is not set. Put it in ~/.Renviron and restart R.")
  body <- toJSON(list(model = MODEL_ID,
                      messages = list(list(role = "user", content = prompt_text)),
                      temperature = 0.4, max_tokens = 6000, reasoning_effort = "low",
                      response_format = list(type = "json_object")), auto_unbox = TRUE)
  for (attempt in 1:5) {
    resp <- tryCatch(httr::POST(ENDPOINT, body = body, encode = "raw",
                                httr::add_headers(Authorization = paste("Bearer", key),
                                                  `Content-Type` = "application/json")),
                     error = function(e) { cat("  [connection error]", conditionMessage(e), "\n"); NULL })
    if (is.null(resp)) { Sys.sleep(5); next }
    sc  <- httr::status_code(resp)
    txt <- httr::content(resp, "text", encoding = "UTF-8")
    if (sc == 200) {
      out <- tryCatch(fromJSON(txt, simplifyVector = FALSE)$choices[[1]]$message$content,
                      error = function(e) NULL)
      if (length(out) && nzchar(trimws(out))) return(out)
      cat("  [empty answer]\n")
    } else if (sc %in% c(401, 403, 404)) {
      stop("Groq returned HTTP ", sc, " (bad/revoked key or wrong model): ", substr(txt, 1, 300))
    } else if (sc == 429) {
      if (grepl("tokens per day|TPD", txt, ignore.case = TRUE)) return("TPD_LIMIT")
      cat("  [rate limit] waiting 15s\n"); Sys.sleep(15)
    } else {
      cat("  [HTTP", sc, "]", substr(txt, 1, 200), "\n"); Sys.sleep(3)
    }
  }
  NA_character_
}

# --------------------------------------------------------------------------
# 4. Parse + validate one batch answer -> one row of probabilities per student
# --------------------------------------------------------------------------
P_GPA   <- paste0("p_cgpa_", 1:8)
P_SCORE <- paste0("p_score_", 1:10)

parse_batch <- function(txt, ids) {
  if (is.na(txt)) return(NULL)
  json <- regmatches(txt, regexpr("\\{[\\s\\S]*\\}", txt, perl = TRUE))
  js <- if (length(json)) tryCatch(fromJSON(json, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(js$students)) return(NULL)
  rows <- lapply(js$students, function(s) {
    g  <- suppressWarnings(as.numeric(unlist(s$cgpa)))
    sc <- suppressWarnings(as.numeric(unlist(s$score)))
    if (is.null(s$id) || length(g) != 8 || length(sc) != 10 || anyNA(c(g, sc)) ||
        any(c(g, sc) < 0) || sum(g) <= 0 || sum(sc) <= 0) return(NULL)
    r <- data.frame(student_id = as.character(s$id), sum_cgpa_raw = sum(g), sum_score_raw = sum(sc))
    r[P_GPA] <- as.list(g / sum(g)); r[P_SCORE] <- as.list(sc / sum(sc))
    r
  })
  P <- do.call(rbind, rows)
  if (is.null(P)) return(NULL)
  P[P$student_id %in% ids & !duplicated(P$student_id), ]
}

batch_ok <- function(P, ids) !is.null(P) && setequal(P$student_id, ids) &&
  (length(ids) == 1 || nrow(unique(round(P[c(P_GPA, P_SCORE)], 3))) > 1)

# --------------------------------------------------------------------------
# 5. Run all batches, resuming from the cache
# --------------------------------------------------------------------------
batches  <- split(split$ai_ids, ceiling(seq_along(split$ai_ids) / BATCH_SIZE))
f_cache  <- out_file("llm_responses")
cache    <- if (file.exists(f_cache)) readRDS(f_cache) else vector("list", length(batches))

for (b in seq_along(batches)) {
  ids <- batches[[b]]
  if (!is.null(cache[[b]]) && identical(cache[[b]]$ids, ids) && cache[[b]]$ok) next
  d <- data_ai_raw[match(ids, data_ai_raw$student_id), ]
  best <- list(ids = ids, text = NA_character_, ok = FALSE, n_valid = 0, tries = 0)
  for (try in seq_len(MAX_TRIES)) {
    cat(sprintf("Batch %d/%d, try %d ...\n", b, length(batches), try))
    txt <- get_response(create_batch_prompt(d))
    if (identical(txt, "TPD_LIMIT")) {
      saveRDS(cache, f_cache)
      stop("Daily token limit reached. Progress saved in ", f_cache, "; re-run tomorrow to resume.")
    }
    P  <- parse_batch(txt, ids)
    nv <- if (is.null(P)) 0 else nrow(P)
    if (nv >= best$n_valid) best <- list(ids = ids, text = txt, ok = batch_ok(P, ids),
                                         n_valid = nv, tries = try)
    if (best$ok) break
    cat("  incomplete or degenerate answer (", nv, "/", length(ids), "valid ) - retrying\n")
  }
  cache[[b]] <- best
  saveRDS(cache, f_cache)
  if (!DRY_RUN) Sys.sleep(2.5)
}

# --------------------------------------------------------------------------
# 6. Probabilities, sampled labels (as in bapr2.R), and the AI dataset
# --------------------------------------------------------------------------
probs <- do.call(rbind, lapply(cache, function(cb) parse_batch(cb$text, cb$ids)))
probs <- probs[match(split$ai_ids, probs$student_id), ]
probs$student_id <- split$ai_ids                         # unmatched students stay as NA rows
probs$exp_cgpa  <- as.vector(as.matrix(probs[P_GPA])   %*% ((GPA_LO + GPA_HI) / 2))
probs$exp_score <- as.vector(as.matrix(probs[P_SCORE]) %*% ((SCORE_LO + SCORE_HI) / 2))
saveRDS(probs, out_file("llm_probs"))

set.seed(SEED)
draw <- function(p, lo, hi) {
  if (anyNA(p)) return(NA_real_)
  k <- sample(length(p), 1, prob = p); runif(1, lo[k], hi[k])
}
data_ai_complete <- data_ai_raw
data_ai_complete$cgpa <- round(apply(probs[P_GPA], 1, draw, GPA_LO, GPA_HI), 2)
data_ai_complete$critical_thinking_score <- round(apply(probs[P_SCORE], 1, draw, SCORE_LO, SCORE_HI), 1)
saveRDS(data_ai_complete, out_file("data_ai_complete"))
saveRDS(data_known, out_file("data_known"))

# --------------------------------------------------------------------------
# 7. Quality report (true labels used ONLY here, for evaluation)
# --------------------------------------------------------------------------
truth <- data[match(split$ai_ids, data$student_id), ]
ok    <- !is.na(probs$exp_cgpa)
one_hot <- function(cols) sum(apply(probs[ok, cols], 1, max) > 0.99)
cat("\n===== LLM label quality (", OUT_TAG, ") =====\n")
cat("Batches OK:", sum(vapply(cache, function(cb) isTRUE(cb$ok), TRUE)), "/", length(batches),
    " | students with valid answers:", sum(ok), "/", N_AI, "\n")
cat("Raw prob sums off by >5%:", sum(abs(probs$sum_cgpa_raw[ok] - 1) > 0.05 |
                                     abs(probs$sum_score_raw[ok] - 1) > 0.05),
    " | one-hot cgpa:", one_hot(P_GPA), " one-hot score:", one_hot(P_SCORE), "\n")
qual <- data.frame(
  outcome    = OUTCOMES,
  mean_llm   = c(mean(probs$exp_cgpa[ok]), mean(probs$exp_score[ok])),
  mean_true  = c(mean(truth$cgpa[ok]), mean(truth$critical_thinking_score[ok])),
  sd_llm     = c(sd(probs$exp_cgpa[ok]), sd(probs$exp_score[ok])),
  sd_true    = c(sd(truth$cgpa[ok]), sd(truth$critical_thinking_score[ok])),
  cor        = c(cor(probs$exp_cgpa[ok], truth$cgpa[ok]),
                 cor(probs$exp_score[ok], truth$critical_thinking_score[ok])),
  mse_llm    = c(mean((probs$exp_cgpa[ok] - truth$cgpa[ok])^2),
                 mean((probs$exp_score[ok] - truth$critical_thinking_score[ok])^2)),
  mse_mean   = c(var(truth$cgpa[ok]), var(truth$critical_thinking_score[ok])))
print(qual, digits = 3, row.names = FALSE)
cat("(mse_mean = error of always predicting the true mean; mse_llm should be below it)\n")
cat("Outputs:", paste(basename(c(f_split, f_cache, out_file("llm_probs"),
                                 out_file("data_ai_complete"), out_file("data_known"))),
                      collapse = ", "), "in", normalizePath(OUT_DIR), "\n")
