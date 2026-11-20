# ==========================================================================
# data_check.R -- STEP 1: is campus_ai_pulse_2026.csv fit for the
# linear-regression half of the AI-prior project (O'Hagan & Rockova 2025)?
#
# Read-only: does NOT modify bapr2.R or any data file.
# Writes plots + a summary table to ./data_check_output/
# Dependencies: base R (+ jsonlite, only for the LLM-response checks).
#
# Usage: put campus_ai_pulse_2026.csv (and optionally risposte_groq_oss20b.rds,
# data_ai_campus_complete.rds) in data/llm/, then from the project root:
#        source("regression/data_check.R")
# ==========================================================================

DATA_DIR <- Sys.getenv("CAMPUS_DATA_DIR", "data")                 # folder holding the csv
LLM_DIR  <- Sys.getenv("LLM_DIR", file.path(DATA_DIR, "llm"))     # folder holding the LLM .rds files
OUT_DIR  <- Sys.getenv("CHECK_OUT_DIR", file.path("regression", "output", "data_check"))
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

Y_SCORE <- "critical_thinking_score"
Y_GPA   <- "cgpa"
X_KEY   <- "ai_hours_per_week"
DROPPED <- c(1, 9, 13, 14, 19:24)                 # columns removed in bapr2.R

csv_file <- file.path(DATA_DIR, "campus_ai_pulse_2026.csv")
if (!file.exists(csv_file)) stop("Dataset not found: ", normalizePath(csv_file, mustWork = FALSE),
                                 "\nSet CAMPUS_DATA_DIR or copy the csv into the project folder.")

checks <- data.frame(check = character(), result = character(),
                     problem = character(), fix = character(), stringsAsFactors = FALSE)
add_check <- function(check, result, problem, fix = "") {
  checks[nrow(checks) + 1, ] <<- list(check, result, problem, fix)
  cat(sprintf("\n[%s] %s -> %s\n", problem, check, result))
}
section <- function(txt) cat("\n\n==========", txt, "==========\n")
skew <- function(x) { x <- x[!is.na(x)]; mean((x - mean(x))^3) / sd(x)^3 }
kurt <- function(x) { x <- x[!is.na(x)]; mean((x - mean(x))^4) / sd(x)^4 - 3 }

# --------------------------------------------------------------------------
section("A. SOURCE, COLUMNS, LEAKAGE")
# --------------------------------------------------------------------------
raw_chr <- read.csv(csv_file, stringsAsFactors = FALSE, colClasses = "character",
                    na.strings = character(0))            # untouched text, for code checks
raw     <- read.csv(csv_file, stringsAsFactors = TRUE)    # exactly as bapr2.R reads it

col_info <- data.frame(idx = seq_along(raw), name = names(raw),
                       class = sapply(raw, function(v) class(v)[1]),
                       n_unique = sapply(raw, function(v) length(unique(v))),
                       dropped_in_bapr2 = seq_along(raw) %in% DROPPED,
                       example = sapply(raw_chr, function(v) paste(head(unique(v), 3), collapse = " / ")),
                       row.names = NULL)
print(col_info, right = FALSE)
write.csv(col_info, file.path(OUT_DIR, "columns.csv"), row.names = FALSE)

for (v in c(Y_SCORE, Y_GPA, X_KEY))
  if (!v %in% names(raw)) stop("Expected column missing: ", v)
if (any(c(Y_SCORE, Y_GPA, X_KEY) %in% names(raw)[DROPPED]))
  add_check("bapr2 drop list", "drops an outcome or the key predictor!", "YES", "fix column indices")
add_check("Columns dropped by bapr2.R",
          paste(names(raw)[DROPPED], collapse = ", "), "CHECK",
          "drop by NAME, not position, and document why each is dropped")

# Leakage: how well does each column (kept or dropped) predict each outcome on its own?
leak <- do.call(rbind, lapply(setdiff(names(raw), c(Y_SCORE, Y_GPA)), function(v) {
  x <- raw[[v]]
  if (is.factor(x) && nlevels(x) > 0.5 * nrow(raw)) return(NULL)   # ID-like
  r2 <- function(y) summary(lm(raw[[y]] ~ x))$r.squared
  data.frame(column = v, dropped = v %in% names(raw)[DROPPED],
             R2_score = round(r2(Y_SCORE), 3), R2_cgpa = round(r2(Y_GPA), 3))
}))
leak <- leak[order(-pmax(leak$R2_score, leak$R2_cgpa)), ]
print(leak, row.names = FALSE)
write.csv(leak, file.path(OUT_DIR, "leakage_r2.csv"), row.names = FALSE)
sus <- leak[pmax(leak$R2_score, leak$R2_cgpa) > 0.5, ]
add_check("Leakage (single-column R2 > 0.5 for an outcome)",
          if (nrow(sus)) paste(sprintf("%s%s", sus$column, ifelse(sus$dropped, "(dropped)", "(KEPT)")), collapse = ", ") else "none",
          if (any(!sus$dropped)) "YES" else if (nrow(sus)) "OK (dropped)" else "no",
          if (any(!sus$dropped)) "remove kept columns derived from the outcome" else "")
cor_outcomes <- cor(raw[[Y_SCORE]], raw[[Y_GPA]])
add_check("cor(score, cgpa)", round(cor_outcomes, 3),
          if (abs(cor_outcomes) > 0.8) "YES" else "no",
          "bapr2 already excludes the other outcome via '- cgpa' / '- critical_thinking_score'")

# --------------------------------------------------------------------------
section("B. BASIC QUALITY")
# --------------------------------------------------------------------------
data <- raw[, -DROPPED]                       # same as bapr2.R
add_check("Dimensions (raw / after drop)",
          sprintf("%d x %d / %d x %d", nrow(raw), ncol(raw), nrow(data), ncol(data)), "no")

# 3. numeric columns read as factors
num_like <- names(raw_chr)[sapply(names(raw_chr), function(v) {
  x <- trimws(raw_chr[[v]]); x <- x[x != "" & !toupper(x) %in% c("NA", "N/A", "NAN")]
  is.factor(raw[[v]]) && length(x) > 0 && mean(!is.na(suppressWarnings(as.numeric(x)))) > 0.9
})]
add_check("Numeric columns wrongly read as factors", if (length(num_like)) paste(num_like, collapse = ", ") else "none",
          if (length(num_like)) "YES" else "no", "clean the bad codes, then as.numeric()")

# 4. missing / empty / sentinel codes / duplicates
na_codes <- c("", "NA", "N/A", "NAN", "NULL", "UNKNOWN", "NONE", "-", "?", "-1", "-999", "999")
miss <- sapply(raw_chr, function(v) sum(toupper(trimws(v)) %in% na_codes))
miss <- miss[miss > 0]
print(miss)
add_check("Missing / empty / sentinel codes", if (length(miss)) paste(names(miss), miss, sep = "=", collapse = ", ") else "none",
          if (length(miss)) "CHECK" else "no", "inspect; '-1'/'999' may be legitimate values")
dup_rows    <- sum(duplicated(raw))
dup_feat    <- sum(duplicated(raw[, -1]))       # ignoring the (assumed) ID column
dup_id      <- sum(duplicated(raw[[1]]))
add_check("Duplicated rows / rows dup. ignoring col 1 / dup. values in col 1",
          sprintf("%d / %d / %d", dup_rows, dup_feat, dup_id),
          if (dup_rows + dup_feat + dup_id > 0) "YES" else "no", "deduplicate before splitting")

# 5. ranges
num_cols <- names(raw)[sapply(raw, is.numeric)]
rng <- data.frame(column = num_cols,
                  min = sapply(raw[num_cols], min, na.rm = TRUE),
                  median = sapply(raw[num_cols], median, na.rm = TRUE),
                  max = sapply(raw[num_cols], max, na.rm = TRUE),
                  n_neg = sapply(raw[num_cols], function(x) sum(x < 0, na.rm = TRUE)),
                  n_out_4sd = sapply(raw[num_cols], function(x) sum(abs(scale(x)) > 4, na.rm = TRUE)),
                  row.names = NULL)
print(rng)
write.csv(rng, file.path(OUT_DIR, "ranges.csv"), row.names = FALSE)
bad_gpa   <- sum(raw[[Y_GPA]] < 0 | raw[[Y_GPA]] > 4, na.rm = TRUE)
bad_score <- sum(raw[[Y_SCORE]] < 0 | raw[[Y_SCORE]] > 100, na.rm = TRUE)
bad_hours <- sum(raw[[X_KEY]] < 0 | raw[[X_KEY]] > 112, na.rm = TRUE)   # >16 h/day
add_check("Impossible values: cgpa outside 0-4 / score outside 0-100 / ai_hours <0 or >112",
          sprintf("%d / %d / %d", bad_gpa, bad_score, bad_hours),
          if (bad_gpa + bad_score + bad_hours) "YES" else "no", "remove or cap")
if ("age" %in% names(raw)) add_check("Age range", paste(range(raw$age, na.rm = TRUE), collapse = "-"),
                                     if (any(raw$age < 15 | raw$age > 80, na.rm = TRUE)) "YES" else "no")

# 6. distributions + 'synthetic fingerprints'
png(file.path(OUT_DIR, "distributions.png"), width = 1500, height = 500)
par(mfrow = c(1, 3))
for (v in c(Y_SCORE, Y_GPA, X_KEY)) {
  hist(raw[[v]], breaks = 40, main = v, xlab = v, col = "grey80", freq = FALSE)
  curve(dnorm(x, mean(raw[[v]]), sd(raw[[v]])), add = TRUE, col = "red", lwd = 2)
}
dev.off()
shape <- do.call(rbind, lapply(c(Y_SCORE, Y_GPA, X_KEY), function(v) {
  x <- raw[[v]]
  dec <- nchar(sub("^[^.]*\\.?", "", format(x, scientific = FALSE, trim = TRUE)))
  data.frame(variable = v, mean = mean(x), sd = sd(x), skew = round(skew(x), 2),
             ex_kurt = round(kurt(x), 2), n_unique = length(unique(x)),
             max_decimals = max(dec), pct_at_min = round(100 * mean(x == min(x)), 1),
             pct_at_max = round(100 * mean(x == max(x)), 1),
             ks_normal_p = signif(suppressWarnings(ks.test(scale(x), "pnorm")$p.value), 3),
             ks_uniform_p = signif(suppressWarnings(ks.test(x, "punif", min(x), max(x))$p.value), 3))
}))
print(shape, row.names = FALSE)
write.csv(shape, file.path(OUT_DIR, "shape.csv"), row.names = FALSE)
ceil <- shape$pct_at_min > 5 | shape$pct_at_max > 5
add_check("Ceiling / floor (>5% at min or max)", paste(shape$variable[ceil], collapse = ", "),
          if (any(ceil)) "YES" else "no", "consider censored/Tobit model or a transformation")
synth <- shape$variable[shape$ks_normal_p > 0.5 | shape$ks_uniform_p > 0.5]
add_check("Suspiciously perfect normal/uniform shape (KS p > 0.5)",
          if (length(synth)) paste(synth, collapse = ", ") else "none",
          if (length(synth)) "CHECK (synthetic?)" else "no")

# 7. rare factor levels
fac_cols <- names(data)[sapply(data, is.factor)]
rare <- unlist(lapply(fac_cols, function(v) {
  tb <- table(data[[v]]); r <- tb[tb < 10]
  if (length(r)) paste0(v, ":", names(r), "(", r, ")") else NULL
}))
for (v in fac_cols) { cat("\n", v, ":\n"); print(table(data[[v]])) }
add_check("Rare factor levels (<10 rows) in modelling data",
          if (length(rare)) paste(rare, collapse = ", ") else "none",
          if (length(rare)) "YES" else "no", "merge rare levels; lm() on a 300-row split can fail")
hi_card <- fac_cols[sapply(data[fac_cols], nlevels) > 20]
if (length(hi_card)) add_check("High-cardinality factors (>20 levels)", paste(hi_card, collapse = ", "),
                               "YES", "drop or group; huge number of dummies")

# --------------------------------------------------------------------------
section("C. SUITABILITY FOR LINEAR REGRESSION")
# --------------------------------------------------------------------------
num_d <- names(data)[sapply(data, is.numeric)]
cm <- cor(data[num_d])
print(round(cm, 2))
write.csv(round(cm, 3), file.path(OUT_DIR, "correlations.csv"))
png(file.path(OUT_DIR, "correlations.png"), width = 900, height = 900)
par(mar = c(10, 10, 2, 2))
image(1:ncol(cm), 1:ncol(cm), cm[, ncol(cm):1], zlim = c(-1, 1), axes = FALSE, xlab = "", ylab = "",
      col = hcl.colors(41, "Blue-Red 3"), main = "Correlations (numeric, kept columns)")
axis(1, 1:ncol(cm), colnames(cm), las = 2); axis(2, 1:ncol(cm), rev(colnames(cm)), las = 2)
text(rep(1:ncol(cm), ncol(cm)), rep(ncol(cm):1, each = ncol(cm)), sprintf("%.2f", cm), cex = 0.8)
dev.off()

# 8. VIF (generalised: 1/(1-R2) of each predictor on the others, numeric predictors)
preds <- setdiff(names(data), c(Y_SCORE, Y_GPA))
vif <- sapply(intersect(preds, num_d), function(v) {
  f <- reformulate(setdiff(preds, v), response = v)
  1 / (1 - summary(lm(f, data = data))$r.squared)
})
print(round(sort(vif, decreasing = TRUE), 2))
add_check("Max VIF (numeric predictors, given all others)",
          sprintf("%s = %.2f", names(which.max(vif)), max(vif)),
          if (max(vif) > 5) "YES" else "no", "drop/merge collinear predictors")

# 9-10. OLS for both outcomes
fit_models <- list(
  score_all  = lm(reformulate(setdiff(names(data), c(Y_SCORE, Y_GPA)), Y_SCORE), data = data),
  score_only = lm(reformulate(X_KEY, Y_SCORE), data = data),
  cgpa_all   = lm(reformulate(setdiff(names(data), c(Y_SCORE, Y_GPA)), Y_GPA), data = data),
  cgpa_only  = lm(reformulate(X_KEY, Y_GPA), data = data))
ols <- do.call(rbind, lapply(names(fit_models), function(m) {
  f <- fit_models[[m]]; s <- summary(f); ci <- confint(f)[X_KEY, ]
  data.frame(model = m, n = nobs(f), p = length(coef(f)) - 1,
             R2 = round(s$r.squared, 3), adjR2 = round(s$adj.r.squared, 3),
             sigma = signif(s$sigma, 3),
             beta_ai = signif(coef(f)[X_KEY], 3), ci_lo = signif(ci[1], 3), ci_hi = signif(ci[2], 3),
             t_ai = round(coef(s)[X_KEY, "t value"], 2),
             partial_R2_ai = round(coef(s)[X_KEY, "t value"]^2 /
                                   (coef(s)[X_KEY, "t value"]^2 + f$df.residual), 4),
             max_cook = round(max(cooks.distance(f), na.rm = TRUE), 3),
             n_cook_gt_4n = sum(cooks.distance(f) > 4 / nobs(f), na.rm = TRUE),
             shapiro_p = signif(shapiro.test(head(resid(f), 5000))$p.value, 3))
}))
print(ols, row.names = FALSE)
write.csv(ols, file.path(OUT_DIR, "ols_full_data.csv"), row.names = FALSE)
for (m in names(fit_models)) {
  png(file.path(OUT_DIR, paste0("residuals_", m, ".png")), width = 1000, height = 1000)
  par(mfrow = c(2, 2)); plot(fit_models[[m]], which = c(1, 2, 3, 4), main = m); dev.off()
}
for (i in seq_len(nrow(ols))) with(ols[i, ], add_check(
  sprintf("OLS %s: R2, beta_ai [95%% CI]", model),
  sprintf("R2=%.3f, beta=%s [%s, %s], partial R2(ai)=%.4f", R2, beta_ai, ci_lo, ci_hi, partial_R2_ai),
  if (R2 < 0.05 || (ci_lo < 0 & ci_hi > 0)) "YES (weak signal)" else "no",
  if (R2 < 0.05) "truth is ~noise: an LLM prior can only add bias; consider a different outcome/dataset" else ""))

# 11. sample size: how precise is beta_ai at the planned n=300 labelled?
n_tot <- nrow(data)
se_at <- function(n, f) summary(f)$coef[X_KEY, "Std. Error"] * sqrt(nobs(f) / n)
ss <- do.call(rbind, lapply(names(fit_models), function(m) {
  f <- fit_models[[m]]; b <- coef(f)[X_KEY]
  data.frame(model = m, beta_full = signif(b, 3),
             se_n50 = signif(se_at(50, f), 3), se_n100 = signif(se_at(100, f), 3),
             se_n300 = signif(se_at(300, f), 3),
             z_at_n300 = round(b / se_at(300, f), 2))
}))
print(ss, row.names = FALSE)
write.csv(ss, file.path(OUT_DIR, "sample_size.csv"), row.names = FALSE)
add_check("Rows available vs 300 known + 900 AI",
          sprintf("%d rows; 1200 used; %d unused", n_tot, n_tot - 1200),
          if (n_tot < 1200) "YES" else "no",
          "smaller n_known (50-100) makes the prior's effect visible; keep a held-out test set")

# --------------------------------------------------------------------------
section("D. LLM-PRIOR CHECKS (only if files exist)")
# --------------------------------------------------------------------------
f_resp <- file.path(LLM_DIR, "risposte_groq_oss20b.rds")
f_comp <- file.path(LLM_DIR, "data_ai_campus_complete.rds")

if (file.exists(f_resp) && requireNamespace("jsonlite", quietly = TRUE)) {
  resp <- readRDS(f_resp)
  per_batch <- t(sapply(seq_along(resp), function(b) {
    r <- resp[b]
    if (is.na(r)) return(c(json_ok = 0, n_students = 0, n_valid = 0, bad_sum = 0))
    txt <- regmatches(r, regexpr("\\{[\\s\\S]*\\}", r, perl = TRUE))
    js <- if (length(txt)) tryCatch(jsonlite::fromJSON(txt, simplifyVector = FALSE), error = function(e) NULL)
    if (is.null(js$students)) return(c(json_ok = 0, n_students = 0, n_valid = 0, bad_sum = 0))
    ok <- sapply(js$students, function(s) length(unlist(s$cgpa)) == 8 && length(unlist(s$score)) == 10)
    sums <- unlist(lapply(js$students[ok], function(s) c(sum(unlist(s$cgpa)), sum(unlist(s$score)))))
    c(json_ok = 1, n_students = length(js$students), n_valid = sum(ok),
      bad_sum = sum(abs(sums - 1) > 0.05))
  }))
  print(colSums(per_batch))
  add_check("LLM responses: batches NA / bad JSON / students parsed / valid 8+10 bins / prob sums off by >0.05",
            sprintf("%d / %d / %d / %d / %d", sum(is.na(resp)), sum(per_batch[, "json_ok"] == 0 & !is.na(resp)),
                    sum(per_batch[, "n_students"]), sum(per_batch[, "n_valid"]), sum(per_batch[, "bad_sum"])),
            if (any(per_batch[, "json_ok"] == 0) || any(per_batch[, "n_students"] != 15)) "YES" else "no",
            "use response_format json_object; match students by 'id', not position")
}

if (file.exists(f_comp)) {
  ai <- readRDS(f_comp)
  # Recover the TRUE labels of the rows the LLM labelled, by matching on all feature columns.
  feats <- setdiff(names(ai), c(Y_SCORE, Y_GPA))
  key  <- function(d) do.call(paste, c(lapply(d[feats], as.character), sep = "|"))
  truth_idx <- match(key(ai), key(data))
  add_check("AI rows matched back to original data", sprintf("%d / %d (ambiguous dup keys: %d)",
            sum(!is.na(truth_idx)), nrow(ai), sum(duplicated(key(data)))),
            if (anyNA(truth_idx)) "YES" else "no")
  add_check("NA in LLM-imputed labels (cgpa / score)",
            sprintf("%d / %d", sum(is.na(ai[[Y_GPA]])), sum(is.na(ai[[Y_SCORE]]))),
            if (anyNA(ai[[Y_GPA]]) || anyNA(ai[[Y_SCORE]])) "YES" else "no", "lm() silently drops them")
  cmp <- do.call(rbind, lapply(c(Y_SCORE, Y_GPA), function(y) {
    a <- ai[[y]]; t <- data[[y]][truth_idx]; ok <- !is.na(a) & !is.na(t)
    data.frame(outcome = y, mean_llm = mean(a[ok]), mean_true = mean(t[ok]),
               sd_llm = sd(a[ok]), sd_true = sd(t[ok]),
               cor = round(cor(a[ok], t[ok]), 3), mse = mean((a[ok] - t[ok])^2),
               mse_predict_mean = mean((mean(t[ok]) - t[ok])^2))
  }))
  print(cmp, row.names = FALSE)
  write.csv(cmp, file.path(OUT_DIR, "llm_vs_true.csv"), row.names = FALSE)
  for (i in seq_len(nrow(cmp))) with(cmp[i, ], add_check(
    sprintf("LLM vs true %s: cor, MSE (vs MSE of predicting the mean)", outcome),
    sprintf("cor=%.3f, MSE=%.3g (baseline %.3g), mean %.3g vs %.3g, sd %.3g vs %.3g",
            cor, mse, mse_predict_mean, mean_llm, mean_true, sd_llm, sd_true),
    if (cor < 0.2 || mse > mse_predict_mean) "YES" else "no",
    "LLM labels carry little information; prior will mostly add noise/bias"))
  png(file.path(OUT_DIR, "llm_vs_true.png"), width = 1000, height = 500)
  par(mfrow = c(1, 2))
  for (y in c(Y_SCORE, Y_GPA)) { plot(data[[y]][truth_idx], ai[[y]], xlab = "true", ylab = "LLM-imputed",
                                      main = y, pch = 16, col = "#00000040"); abline(0, 1, col = "red") }
  dev.off()
}
add_check("Split reproducibility in bapr2.R",
          "sample() is called before set.seed(30175)", "YES",
          "move set.seed() above the shuffle; otherwise cached LLM labels get attached to different rows on every run")

# --------------------------------------------------------------------------
section("SUMMARY TABLE")
# --------------------------------------------------------------------------
print(checks, right = FALSE, row.names = FALSE)
write.csv(checks, file.path(OUT_DIR, "summary_checks.csv"), row.names = FALSE)
cat("\nOutputs written to", normalizePath(OUT_DIR), "\n")
