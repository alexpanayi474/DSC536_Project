# 03_llm_prior.R: LLM class probabilities for the AI prior (classification half).
# Follows the paper's skin disease example (O'Hagan & Rockova 2025, Sec. 5.1.2
# and Appendix A.1): one patient per call, a prompt that explains every feature
# and gives a clinical profile of each class, answer after an !ANSWER! delimiter.
#
# Input : data/pad_clean.csv (which patients, from 01_clean.R)
#         data/metadata.csv  (raw feature values, used to write the case text)
# Output: data/llm_classification/llm_responses.rds  raw answers and token usage
#         data/llm_classification/llm_probs.csv      one row per patient: patient_id,
#                                                    lesion_id, p_ACK ... p_SEK, llm_pred
#         data/llm_classification/llm_prompt.txt     the exact system prompt used
#
# The probabilities are kept in a separate table keyed by (patient_id, lesion_id)
# and joined to pad_clean.csv when needed. LLM answers are expensive and fixed,
# while the cleaning may change; a separate table also keeps p_* out of the
# feature matrix by construction.
#
# The prompt uses only the variable definitions of the dataset and general
# clinical descriptions of the six diagnoses. Nothing in it is taken from the
# relationship between features and diagnoses in our data, so the prior is a
# prior. Fix the prompt before the full run and do not change it afterwards.
#
# Setup: ANTHROPIC_API_KEY in ~/.Renviron, renv::install("httr2").
# Run from the project root: source("classification/03_llm_prior.R")
#   Pilot on a few patients:  Sys.setenv(LLM_PILOT = "20")
#   Test without API calls:   Sys.setenv(LLM_DRY_RUN = "1")  (writes to a separate dryrun folder)
# The script resumes from the cache, so running it again only sends the
# patients without a valid answer.

library(httr2)
if (packageVersion("httr2") < "1.1.0") stop("httr2 >= 1.1.0 is needed: renv::install(\"httr2\")")

DATA_DIR  <- "data"
DRY_RUN   <- identical(Sys.getenv("LLM_DRY_RUN"), "1")
LLM_DIR   <- file.path(DATA_DIR, if (DRY_RUN) "llm_classification_dryrun" else "llm_classification")
CLEAN     <- file.path(DATA_DIR, "pad_clean.csv")
RAW       <- file.path(DATA_DIR, "metadata.csv")
F_CACHE   <- file.path(LLM_DIR, "llm_responses.rds")
F_PROBS   <- file.path(LLM_DIR, "llm_probs.csv")
F_PROMPT  <- file.path(LLM_DIR, "llm_prompt.txt")

ENDPOINT   <- "https://api.anthropic.com/v1/messages"
API_VER    <- "2026-10-21"
MODEL_ID   <- "claude-haiku-5-5"
MAX_TOKENS <- 2000
PARALLEL   <- 8            # requests in flight at once; lower it on many 429 errors
CHUNK      <- 40           # cache is saved after every chunk
PILOT      <- suppressWarnings(as.integer(Sys.getenv("LLM_PILOT", "")))

CLASSES    <- c(ACK = "actinic keratosis", BCC = "basal cell carcinoma",
                SCC = "squamous cell carcinoma", MEL = "melanoma",
                NEV = "nevus", SEK = "seborrheic keratosis")
P_COLS     <- paste0("p_", names(CLASSES))
FEATURES   <- c("age", "gender", "fitspatrick", "smoke", "drink", "pesticide",
                "skin_cancer_history", "cancer_history", "has_piped_water",
                "has_sewage_system", "background_father", "background_mother",
                "region", "diameter_1", "diameter_2",
                "itch", "grew", "hurt", "changed", "bleed", "elevation")

dir.create(LLM_DIR, showWarnings = FALSE, recursive = TRUE)
key_of <- function(p, l) paste(p, l, sep = "|")

# --------------------------------------------------------------------------
# 1. Patients and their raw feature values
# --------------------------------------------------------------------------
clean <- read.csv(CLEAN, stringsAsFactors = FALSE)
raw   <- read.csv(RAW, stringsAsFactors = FALSE, na.strings = c("", "NA"))
raw   <- raw[!duplicated(key_of(raw$patient_id, raw$lesion_id)), ]
raw$background_father[raw$background_father == "BRASIL"] <- "BRAZIL"
raw$background_mother[raw$background_mother == "BRASIL"] <- "BRAZIL"

cases <- raw[match(key_of(clean$patient_id, clean$lesion_id),
                   key_of(raw$patient_id, raw$lesion_id)), c("patient_id", "lesion_id", FEATURES)]
stopifnot(nrow(cases) == nrow(clean), !anyNA(cases))
cases$key <- key_of(cases$patient_id, cases$lesion_id)

case_text <- function(i) {
  vals <- vapply(FEATURES, function(v) as.character(cases[[v]][i]), "")
  paste0(
    "The case is described by: ", paste0(FEATURES, ": ", vals, collapse = ", "), "\n\n",
    "The possible classes are: ", paste(CLASSES, collapse = ", "), ".\n\n",
    "Please estimate the probability of each possible diagnosis for this case. The following is ",
    "for research purposes only. I understand that a real patient must see a qualified doctor ",
    "with such a concern.\n\n",
    "Please think carefully about accurate probabilities. Once you have a reasonable answer, ",
    "please say !ANSWER! and then provide your answer strictly in the following format:\n\n",
    paste0(CLASSES, ": (prob)", collapse = ",\n"), "\n\n",
    "Do not include any other text after the answer.")
}

# --------------------------------------------------------------------------
# 2. System prompt (same for every call, so it can be cached)
# --------------------------------------------------------------------------
# Structure of Appendix A.1: task, meaning of every input feature, clinical
# profile of each class, analytical strategy, a hypothetical worked example.
# The profiles are general clinical descriptions; check them against the
# reference cited in the report before the full run.
SYSTEM_PROMPT <- paste(
"You are an advanced AI medical diagnostic assistant. Your current task is to analyze a set of clinical features of a patient and one skin lesion, and estimate the probability of six potential diagnoses: actinic keratosis, basal cell carcinoma, squamous cell carcinoma, melanoma, nevus and seborrheic keratosis.

The data come from a public dermatology programme in southeastern Brazil that serves many rural and agricultural patients. No image of the lesion is available. Your analysis should be based on a deep understanding of how these features typically present in each disease. The input will be a list of features with their values. Pay close attention to the meaning of each feature.

Understanding the Input Features:

Patient features:
age: the patient's age in years.
gender: MALE or FEMALE.
fitspatrick: Fitzpatrick skin type from 1 to 6. 1 is very fair skin that always burns and never tans, 6 is deeply pigmented skin that never burns. Lower values mean higher sensitivity to sun.
smoke: True if the patient smokes cigarettes.
drink: True if the patient drinks alcohol.
pesticide: True if the patient works or has worked with pesticides.
skin_cancer_history: True if the patient or someone in their family has had skin cancer.
cancer_history: True if the patient or someone in their family has had any type of cancer.
has_piped_water: True if the patient's home has piped water.
has_sewage_system: True if the patient's home is connected to a sewage system.
background_father, background_mother: the country or region of origin of the father's and the mother's family. UNK means not known.

Lesion features:
region: the body region where the lesion is located. Possible values: ABDOMEN, ARM, BACK, CHEST, EAR, FACE, FOOT, FOREARM, HAND, LIP, NECK, NOSE, SCALP, THIGH.
diameter_1, diameter_2: the two diameters of the lesion in millimetres.
itch: True if the lesion itches.
grew: True if the lesion has grown recently.
hurt: True if the lesion hurts.
changed: True if the lesion has changed its pattern recently.
bleed: True if the lesion has bled.
elevation: True if the lesion is raised above the skin.
For itch, grew, hurt, changed, bleed and elevation, the value UNK means the information was not recorded. Treat UNK as missing information, not as True or False.

Detailed Disease Profiles and Feature Interpretation Guidelines:

Carefully consider the following characteristics for each disease when evaluating the case. The presence of typical features supports a diagnosis, while their absence, or the presence of contradictory features, should lower its probability.

Actinic Keratosis:
Key Indicators:
  region: chronically sun exposed sites, such as FACE, SCALP, EAR, FOREARM and HAND.
  age: older adults.
  fitspatrick: fair skin (low values).
  diameter: usually small, a few millimetres.
  itch: may itch or sting.
Less Indicative / Contradictory:
  bleed: bleeding is uncommon.
  grew or changed: marked growth suggests progression to squamous cell carcinoma.
  Young age, or location on skin that is rarely exposed to sun.
Note: a precancerous lesion caused by chronic sun damage.

Basal Cell Carcinoma:
Key Indicators:
  region: head and neck, especially NOSE, FACE and EAR.
  age: older adults.
  fitspatrick: fair skin (low values).
  elevation: often a raised, pearly nodule.
  bleed: may ulcerate and bleed.
  grew: grows slowly over months or years.
  skin_cancer_history: a previous skin cancer raises the probability.
Less Indicative / Contradictory:
  hurt: usually painless.
  Young age.
Note: the most common skin cancer.

Squamous Cell Carcinoma:
Key Indicators:
  region: sun exposed sites, such as FACE, EAR, LIP, SCALP, HAND and FOREARM.
  age: older adults.
  fitspatrick: fair skin (low values).
  elevation: a raised, scaly or crusted lesion.
  grew: may grow faster than basal cell carcinoma.
  hurt and bleed: may be tender, painful or bleed.
  skin_cancer_history: a previous skin cancer raises the probability.
Less Indicative / Contradictory:
  Young age, small stable lesions without symptoms.
Note: includes Bowen's disease (squamous cell carcinoma in situ). May arise from an actinic keratosis.

Melanoma:
Key Indicators:
  grew and changed: recent growth or change in size, shape or colour is the strongest sign.
  diameter: often larger than 6 millimetres.
  region: can appear anywhere, often on the BACK or CHEST, also on the limbs.
  bleed and itch: may bleed or itch.
  fitspatrick: fair skin (low values).
  skin_cancer_history: a previous skin cancer raises the probability.
Less Indicative / Contradictory:
  A small, stable lesion that has not grown or changed.
Note: cancer of the pigment cells. Less common than the other cancers but can occur at a younger age.

Nevus:
Key Indicators:
  age: more frequent in children and younger adults.
  region: often on the trunk and limbs, such as BACK, CHEST, ARM and THIGH.
  diameter: usually small.
  grew and changed: usually stable in adults.
Less Indicative / Contradictory:
  bleed, hurt, recent growth or change in an older patient.
Note: a common benign mole.

Seborrheic Keratosis:
Key Indicators:
  age: middle aged and older adults.
  elevation: raised, with a stuck on, waxy or warty appearance.
  region: often on the trunk (BACK, CHEST) and the FACE.
  itch: may itch.
Less Indicative / Contradictory:
  bleed and hurt are uncommon unless irritated.
  Young age.
Note: a very common benign growth, not caused by sun damage.

General Analytical Strategy for Probability Estimation:

Holistic Review: do not assess features in isolation. Consider the entire clinical picture.
Key Features: give significant weight to the hallmark features of each disease as described in its profile.
Consistency: check whether the features fit together. For example, a lesion on the nose of an older patient with fair skin fits sun related lesions better than a nevus.
Differential Diagnosis: actively consider why it might be one disease and not another. For example, an older patient with a small lesion on the forearm could be an actinic keratosis or a squamous cell carcinoma; recent growth, pain or bleeding would move the probability towards squamous cell carcinoma.
Missing information: if a feature is UNK, rely on the other features.
Probabilities: the probabilities should reflect your confidence in each diagnosis based on the evidence. They should sum to 1.0. A disease that matches its typical profile with several key features and no contradictions should receive a high probability. A disease with many contradictory findings should receive a very low probability.

Example of applying the logic (hypothetical case, do not use these probabilities for the actual case below):
Consider: 'age: 72, gender: MALE, fitspatrick: 1, skin_cancer_history: True, region: NOSE, diameter_1: 8, diameter_2: 6, itch: False, grew: True, hurt: False, changed: False, bleed: True, elevation: True'

Basal Cell Carcinoma: older patient, very fair skin, lesion on the nose, raised, bleeding, slowly growing, previous skin cancer. High probability.
Squamous Cell Carcinoma: also fits a sun exposed site in an older patient, but the lesion is painless. Moderate probability.
Actinic Keratosis: the location fits, but growth and bleeding are less typical. Low probability.
Melanoma: growth is present, but the location and the other features fit sun related lesions better. Low probability.
Nevus: older patient with a bleeding, growing lesion on the nose. Very low probability.
Seborrheic Keratosis: possible in an older patient, but bleeding and the location are less typical. Very low probability.

Now, proceed with the diagnosis for the case provided below using this detailed framework.")

writeLines(c(paste("# model:", MODEL_ID, "| saved:", format(Sys.time(), "%Y-%m-%d %H:%M")),
             SYSTEM_PROMPT), F_PROMPT)

# --------------------------------------------------------------------------
# 3. API call and parsing
# --------------------------------------------------------------------------
make_req <- function(i) {
  key <- Sys.getenv("ANTHROPIC_API_KEY")
  request(ENDPOINT) |>
    req_headers(Authorization = paste("Bearer", key), `anthropic-version` = API_VER) |>
    req_body_json(list(
      model = MODEL_ID,
      max_tokens = MAX_TOKENS,
      system = list(list(type = "text", text = SYSTEM_PROMPT,
                         cache_control = list(type = "ephemeral"))),
      messages = list(list(role = "user", content = case_text(i))))) |>
    req_retry(max_tries = 4,
              is_transient = function(r) resp_status(r) %in% c(429, 500, 502, 503, 529))
}

# the answer is the text block; a thinking block may come before it
answer_text <- function(body) {
  txt <- vapply(body$content, function(b) if (identical(b$type, "text")) b$text else "", "")
  paste(txt, collapse = "\n")
}

parse_answer <- function(txt) {
  if (is.na(txt) || !grepl("!ANSWER!", txt, fixed = TRUE)) return(NULL)
  ans <- tolower(sub("^.*!ANSWER!", "", txt))
  p <- vapply(CLASSES, function(cl) {
    m <- regmatches(ans, regexec(paste0(cl, "\\s*:\\s*\\(?\\s*([0-9]*\\.?[0-9]+)"), ans))[[1]]
    if (length(m) == 2) as.numeric(m[2]) else NA_real_
  }, 0)
  if (anyNA(p) || any(p < 0) || sum(p) <= 0) return(NULL)
  c(sum_raw = sum(p), setNames(p / sum(p), P_COLS))
}

fake_answer <- function() {
  x <- rgamma(length(CLASSES), 1); x <- round(x / sum(x), 3)
  paste0("dry run\n!ANSWER!\n", paste0(CLASSES, ": ", x, collapse = ",\n"))
}

# --------------------------------------------------------------------------
# 4. Run, resuming from the cache
# --------------------------------------------------------------------------
if (!DRY_RUN && !nzchar(Sys.getenv("ANTHROPIC_API_KEY")))
  stop("ANTHROPIC_API_KEY is not set. Put it in ~/.Renviron and restart R.")

cache <- if (file.exists(F_CACHE) && !DRY_RUN) readRDS(F_CACHE) else list()
todo  <- which(!vapply(cases$key, function(k) isTRUE(cache[[k]]$ok), TRUE))
cat(nrow(cases) - length(todo), "of", nrow(cases), "patients already done. Model:", MODEL_ID, "\n")
if (!is.na(PILOT)) todo <- head(todo, PILOT)
cat("Sending", length(todo), "patients now.\n")

for (chunk in split(todo, ceiling(seq_along(todo) / CHUNK))) {
  if (DRY_RUN) {
    for (i in chunk) cache[[cases$key[i]]] <- list(text = fake_answer(), ok = TRUE, usage = NULL)
    next
  }
  resps <- req_perform_parallel(lapply(chunk, make_req), on_error = "continue",
                                max_active = PARALLEL, progress = FALSE)
  for (j in seq_along(chunk)) {
    k <- cases$key[chunk[j]]; r <- resps[[j]]
    if (inherits(r, "error")) {
      msg <- conditionMessage(r)
      if (grepl("401|403", msg)) stop("Authentication failed: check the API key. ", msg)
      cache[[k]] <- list(text = NA_character_, ok = FALSE, error = msg); next
    }
    body <- resp_body_json(r)
    txt  <- answer_text(body)
    cache[[k]] <- list(text = txt, ok = !is.null(parse_answer(txt)),
                       stop_reason = body$stop_reason, usage = body$usage)
  }
  saveRDS(cache, F_CACHE)
  n_ok <- sum(vapply(cases$key, function(k) isTRUE(cache[[k]]$ok), TRUE))
  cat(sprintf("  %d / %d patients with a valid answer\n", n_ok, nrow(cases)))
}
if (!DRY_RUN) saveRDS(cache, F_CACHE)

# --------------------------------------------------------------------------
# 5. Token usage and cost (Haiku 5.5 list prices per million tokens)
# --------------------------------------------------------------------------
u <- do.call(rbind, lapply(cache, function(x) {
  if (is.null(x$usage)) return(NULL)
  g <- function(f) if (is.null(x$usage[[f]])) 0 else x$usage[[f]]
  c(input = g("input_tokens"), cache_read = g("cache_read_input_tokens"),
    cache_write = g("cache_creation_input_tokens"), output = g("output_tokens"))
}))
if (!is.null(u)) {
  tot  <- colSums(u)
  cost <- sum(tot * c(0.10, 0.01, 0.125, 0.50)) / 1e6
  cat("\nTokens:", paste(names(tot), tot, sep = " = ", collapse = ", "), "\n")
  cat(sprintf("Cost so far: about $%.3f, about $%.4f per patient\n", cost, cost / nrow(u)))
  cut <- sum(vapply(cache, function(x) identical(x$stop_reason, "max_tokens"), TRUE))
  if (cut > 0) cat(cut, "answers were cut by max_tokens; raise MAX_TOKENS and run again.\n")
}

# --------------------------------------------------------------------------
# 6. Probability table
# --------------------------------------------------------------------------
rows  <- lapply(cases$key, function(k) parse_answer(if (is.null(cache[[k]])) NA_character_ else cache[[k]]$text))
n_bad <- sum(vapply(rows, is.null, TRUE))

if (n_bad > 0) {
  cat("\n", n_bad, "patients still have no valid answer. Run the script again to retry only those.\n")
} else {
  probs <- data.frame(patient_id = cases$patient_id, lesion_id = cases$lesion_id,
                      do.call(rbind, rows), check.names = FALSE)
  probs$llm_pred <- names(CLASSES)[max.col(as.matrix(probs[P_COLS]), ties.method = "first")]
  write.csv(probs, F_PROBS, row.names = FALSE)

  # evaluation only: the true labels are used here to describe the prior,
  # never to change the prompt
  truth <- clean$diagnostic[match(key_of(probs$patient_id, probs$lesion_id),
                                  key_of(clean$patient_id, clean$lesion_id))]
  cat("\n===== LLM prior quality (", MODEL_ID, ") =====\n", sep = "")
  cat(sprintf("Accuracy of the most probable class: %.3f\n", mean(probs$llm_pred == truth)))
  cat(sprintf("Mean probability given to the true class: %.3f\n",
              mean(as.matrix(probs[P_COLS])[cbind(seq_along(truth), match(truth, names(CLASSES)))])))
  cat("Confusion (rows = true, columns = LLM):\n")
  print(table(true = factor(truth, names(CLASSES)), llm = factor(probs$llm_pred, names(CLASSES))))
  cat("Answers with a probability above 0.99 for one class:", sum(apply(probs[P_COLS], 1, max) > 0.99), "\n")
  cat("Saved:", F_PROBS, "\n")
}
