# 01_clean.R: preprocessing of PAD-UFES-20 for the classification half
# of the AI-prior project (O'Hagan & Rockova 2025, Section 5.1).
#
# Input : data/metadata.csv (one row per image)
# Output: data/pad_clean.csv (one row per patient, one-hot encoded)
#
# Steps
# 1. Drop img_id (file name) and biopsed (every cancer is biopsied, so it
#    gives away the class).
# 2. One row per lesion: images of the same lesion share all metadata.
#    lesion_id is not unique on its own, the key is (patient_id, lesion_id).
# 3. Print missingness before cleaning. The patient variables are missing
#    together, only for ACK, NEV and SEK, and only for lesions that were not
#    biopsied. Missingness therefore depends on the class.
# 4. Keep complete cases. Imputing would let the model recognise imputed
#    rows as benign (leakage), so incomplete lesions are dropped.
# 5. One lesion per patient, chosen at random. Lesions of one patient share
#    the patient variables, so keeping several would make rows dependent and
#    could put the same patient in train and test. Independent rows also
#    match the exchangeability assumed by the DP posterior.
# 6. One-hot encode, most frequent level as reference (k - 1 dummies, no
#    collinearity with the intercept). UNK is a real answer ("not known")
#    and gets its own dummy instead of being dropped or imputed.
#
# About 800 patients remain, more than the roughly 300 of the paper's
# example, and enough for a 15% labelled split.
# Numeric columns are not scaled here; scale inside each split with
# training statistics only.
#
# Usage, from the project root: source("classification/01_clean.R")

DATA_DIR <- "data"
IN_FILE  <- file.path(DATA_DIR, "metadata.csv")
OUT_FILE <- file.path(DATA_DIR, "pad_clean.csv")
SEED     <- 536
MIN_LEVEL <- 10        # background levels with fewer patients go to OTHER

if (!file.exists(IN_FILE)) stop("Dataset not found: ", normalizePath(IN_FILE, mustWork = FALSE))

section <- function(txt) cat("\n\n==========", txt, "==========\n")
n_info  <- function(d, label) {
  cat(sprintf("%-38s rows = %4d   patients = %4d\n", label, nrow(d), length(unique(d$patient_id))))
}
class_tab <- function(d) print(table(d$diagnostic))

PATIENT_VARS <- c("smoke", "drink", "background_father", "background_mother", "pesticide",
                  "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
                  "has_sewage_system", "fitspatrick", "diameter_1", "diameter_2")
SYMPTOMS     <- c("itch", "grew", "hurt", "changed", "bleed", "elevation")
LOGICALS     <- c("smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
                  "has_piped_water", "has_sewage_system")

section("1. READ AND DROP img_id, biopsed")
raw <- read.csv(IN_FILE, stringsAsFactors = FALSE, na.strings = c("", "NA"))
n_info(raw, "raw file (one row per image)")

# img_id is a file name; biopsed is TRUE for every cancer, so it leaks the class
d <- raw[, setdiff(names(raw), c("img_id", "biopsed"))]

section("2. ONE ROW PER LESION")
# images of the same lesion share all metadata; lesion_id alone is not unique
key <- paste(d$patient_id, d$lesion_id)
cat("Distinct lesion keys (patient_id, lesion_id):", length(unique(key)), "\n")
cat("lesion_id values used by more than one patient:",
    sum(tapply(d$patient_id, d$lesion_id, function(x) length(unique(x))) > 1), "\n")
d <- d[!duplicated(key), ]
n_info(d, "after dedup (one row per lesion)")
class_tab(d)

section("3. MISSINGNESS BEFORE CLEANING")
na_count <- sapply(d, function(x) sum(is.na(x)))
print(data.frame(n_missing = na_count, pct = round(100 * na_count / nrow(d), 1))[na_count > 0, ])

# the patient variables go missing together
miss_any <- !complete.cases(d[PATIENT_VARS])
miss_all <- rowSums(is.na(d[PATIENT_VARS])) == length(PATIENT_VARS)
cat("\nLesions with any patient variable missing:", sum(miss_any), "\n")
cat("Lesions with all patient variables missing:", sum(miss_all), "\n")

cat("\nMissing (any patient variable) by class:\n")
print(table(diagnostic = d$diagnostic, missing = miss_any))
cat("\nShare missing by class (%):\n")
print(round(100 * tapply(miss_any, d$diagnostic, mean), 1))
cat("\nBiopsy status by missingness (biopsed taken from the raw file):\n")
bio <- raw$biopsed[match(paste(d$patient_id, d$lesion_id), paste(raw$patient_id, raw$lesion_id))]
print(table(biopsed = bio, missing = miss_any))

cat("\nUNK counts in the symptom columns, by class:\n")
print(sapply(SYMPTOMS, function(v) tapply(d[[v]] == "UNK", d$diagnostic, sum)))

section("4. KEEP COMPLETE CASES")
# missingness only occurs in ACK, NEV, SEK, so imputing it would let the model
# learn the class from the fact that a value was filled in
d <- d[!miss_any, ]
n_info(d, "complete cases (one row per lesion)")
class_tab(d)
cat("Missing values left:", sum(is.na(d)), "\n")

section("5. ONE LESION PER PATIENT")
lpp <- table(d$patient_id)
cat("Lesions per patient:\n"); print(table(lpp))
n_cls <- tapply(d$diagnostic, d$patient_id, function(x) length(unique(x)))
cat("Patients with lesions of more than one class:", sum(n_cls > 1), "\n")

# random choice, so no class is favoured when a patient has several lesions
set.seed(SEED)
d <- d[sample(nrow(d)), ]
d <- d[!duplicated(d$patient_id), ]
d <- d[order(d$patient_id, d$lesion_id), ]
n_info(d, "one lesion per patient")
class_tab(d)

section("6. RECODE")
for (v in c("background_father", "background_mother")) {
  d[[v]][d[[v]] == "BRASIL"] <- "BRAZIL"
  tab  <- table(d[[v]])
  rare <- setdiff(names(tab)[tab < MIN_LEVEL], "UNK")
  cat(v, ": levels merged into OTHER:", if (length(rare)) paste(rare, collapse = ", ") else "none", "\n")
  d[[v]][d[[v]] %in% rare] <- "OTHER"
}
for (v in LOGICALS) d[[v]] <- as.integer(d[[v]] == "True")

# most frequent level as reference, so the remaining levels (UNK included)
# each get their own dummy
cat_vars <- c("gender", "region", "background_father", "background_mother", SYMPTOMS)
for (v in cat_vars) {
  tab <- sort(table(d[[v]]), decreasing = TRUE)
  d[[v]] <- factor(d[[v]], levels = names(tab))
  cat(sprintf("%-18s reference = %-10s levels: %s\n", v, names(tab)[1],
              paste(sprintf("%s(%d)", names(tab), tab), collapse = " ")))
}

# grew_UNK and changed_UNK are concentrated in BCC and SCC, so they may act
# as a partial leak; worth comparing models with and without them
cat("\nUNK by class after cleaning (grew and changed: check for leakage):\n")
print(sapply(SYMPTOMS, function(v) tapply(d[[v]] == "UNK", d$diagnostic, sum)))

section("7. ONE-HOT ENCODING AND SAVE")
# Reference categories (no column of their own, all dummies of the variable = 0):
#   gender            FEMALE      dummies: gender_MALE
#   region            FACE        dummies: region_<other site>
#   background_father POMERANIA   dummies: background_father_<other>, incl. UNK and OTHER
#   background_mother POMERANIA   dummies: background_mother_<other>, incl. UNK and OTHER
#   itch              True        dummies: itch_False, itch_UNK
#   grew              True        dummies: grew_False, grew_UNK
#   hurt              False       dummies: hurt_True, hurt_UNK
#   changed           False       dummies: changed_True, changed_UNK
#   bleed             False       dummies: bleed_True, bleed_UNK
#   elevation         True        dummies: elevation_False, elevation_UNK
# Logical columns (smoke, drink, ...) are a single 0/1 column, reference False.
# References are the most frequent level in the cleaning run (seed 536);
# section 6 prints them, check there if the data or seed change.
num_vars <- c("age", "fitspatrick", "diameter_1", "diameter_2")
X <- model.matrix(as.formula(paste("~", paste(cat_vars, collapse = " + "))), data = d)[, -1]
for (v in cat_vars) colnames(X) <- sub(paste0("^", v, "(?!_)"), paste0(v, "_"), colnames(X), perl = TRUE)

out <- data.frame(patient_id = d$patient_id, lesion_id = d$lesion_id,
                  diagnostic = d$diagnostic, d[num_vars], d[LOGICALS], X,
                  check.names = FALSE)
stopifnot(!anyNA(out), !any(duplicated(out$patient_id)))

write.csv(out, OUT_FILE, row.names = FALSE)

section("SUMMARY")
cat("Saved:", OUT_FILE, "\n")
cat("Rows (patients):", nrow(out), "\n")
cat("Feature columns:", ncol(out) - 3, "(numeric", length(num_vars), ", binary", length(LOGICALS),
    ", dummies", ncol(X), ")\n")
cat("Class counts:\n"); class_tab(out)
cat("\nColumns:\n"); print(names(out))
