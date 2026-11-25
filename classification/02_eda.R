# 02_eda.R: simple exploratory analysis of the cleaned PAD-UFES-20 data.
#
# Input : data/pad_clean.csv (from 01_clean.R, one row per patient)
# Output: classification/results/eda/ (plots as png, tables as csv)
#
# What it looks at
# 1. Class balance.
# 2. Numeric features (age, fitspatrick, diameters): summaries and boxplots
#    by class, and their correlations.
# 3. Categorical features: the one-hot columns are turned back into one
#    variable each, then the share of every level within each class is
#    shown as a table and a heatmap.
# 4. A simple test per feature of whether it differs between classes
#    (Kruskal-Wallis for numeric, chi-square for categorical). These are
#    descriptive only, no correction for multiple testing.
#
# Usage, from the project root: source("classification/02_eda.R")

IN_FILE <- file.path("data", "pad_clean.csv")
OUT_DIR <- file.path("classification", "results", "eda")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
set.seed(536)

if (!file.exists(IN_FILE)) stop("Run classification/01_clean.R first: ", IN_FILE, " not found")

section <- function(txt) cat("\n\n==========", txt, "==========\n")
save_tab <- function(x, name) write.csv(x, file.path(OUT_DIR, name), row.names = FALSE)

d <- read.csv(IN_FILE, stringsAsFactors = FALSE, check.names = FALSE)

# classes ordered by size, used in every table and plot
CLASSES <- names(sort(table(d$diagnostic), decreasing = TRUE))
d$diagnostic <- factor(d$diagnostic, levels = CLASSES)

NUM_VARS <- c("age", "fitspatrick", "diameter_1", "diameter_2")
LOGICALS <- c("smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
              "has_piped_water", "has_sewage_system")
# reference level of each one-hot group (see section 7 of 01_clean.R)
REFS <- c(gender = "FEMALE", region = "FACE", background_father = "POMERANIA",
          background_mother = "POMERANIA", itch = "True", grew = "True", hurt = "False",
          changed = "False", bleed = "False", elevation = "True")

# plot colours: one blue for single series, a light to dark blue ramp for heatmaps
BLUE <- "#257de0"
RAMP <- colorRampPalette(c("#ffffff", "#cce3fd", "#84b9f3", "#338ced", "#1b64be", "#0e3d7e"))(100)
INK  <- "#191919"
GREY <- "#666666"

section("0. DATA CHECK")
cat("Rows:", nrow(d), "  unique patients:", length(unique(d$patient_id)),
    "  missing values:", sum(is.na(d)), "\n")
stopifnot(!anyNA(d), !any(duplicated(d$patient_id)))

# rebuild one categorical variable per one-hot group
cat_df <- data.frame(row.names = seq_len(nrow(d)))
for (v in names(REFS)) {
  cols <- grep(paste0("^", v, "_"), names(d), value = TRUE)
  M <- as.matrix(d[cols])
  stopifnot(all(rowSums(M) <= 1))
  lev <- sub(paste0("^", v, "_"), "", cols)
  x <- rep(REFS[[v]], nrow(d))
  hit <- rowSums(M) == 1
  x[hit] <- lev[max.col(M[hit, , drop = FALSE])]
  cat_df[[v]] <- factor(x, levels = c(REFS[[v]], lev))
}
for (v in LOGICALS) cat_df[[v]] <- factor(ifelse(d[[v]] == 1, "True", "False"), levels = c("False", "True"))

section("1. CLASS BALANCE")
cc <- table(d$diagnostic)
class_tab <- data.frame(class = names(cc), n = as.integer(cc), pct = round(100 * as.numeric(cc) / nrow(d), 1))
print(class_tab, row.names = FALSE)
save_tab(class_tab, "class_counts.csv")

png(file.path(OUT_DIR, "class_counts.png"), width = 1600, height = 1000, res = 200)
par(mar = c(4, 5, 3, 1), fg = GREY, col.axis = INK, col.lab = INK)
bp <- barplot(rev(class_tab$n), names.arg = rev(class_tab$class), horiz = TRUE, las = 1,
              col = BLUE, border = NA, xlim = c(0, max(class_tab$n) * 1.2),
              xlab = "Patients", main = "Class counts (one lesion per patient)", col.main = INK)
text(rev(class_tab$n), bp, sprintf("%d (%.1f%%)", rev(class_tab$n), rev(class_tab$pct)),
     pos = 4, cex = 0.8, col = INK)
dev.off()

section("2. NUMERIC FEATURES")
num_sum <- function(x) c(n = length(x), mean = mean(x), sd = sd(x), min = min(x),
                         q1 = unname(quantile(x, 0.25)), median = median(x),
                         q3 = unname(quantile(x, 0.75)), max = max(x))
num_tab <- do.call(rbind, lapply(NUM_VARS, function(v) {
  groups <- c(list(All = d[[v]]), split(d[[v]], d$diagnostic))
  out <- t(sapply(groups, num_sum))
  data.frame(variable = v, class = rownames(out), round(out, 2), row.names = NULL)
}))
print(num_tab, row.names = FALSE)
save_tab(num_tab, "numeric_summary.csv")

num_cor <- round(cor(d[NUM_VARS], method = "spearman"), 2)
cat("\nSpearman correlations:\n"); print(num_cor)
save_tab(data.frame(variable = rownames(num_cor), num_cor, row.names = NULL), "numeric_correlation.csv")

png(file.path(OUT_DIR, "numeric_by_class.png"), width = 2000, height = 1600, res = 200)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), fg = GREY, col.axis = INK, col.lab = INK)
labs <- c(age = "Age (years)", fitspatrick = "Fitzpatrick skin type",
          diameter_1 = "Diameter 1 (mm)", diameter_2 = "Diameter 2 (mm)")
for (v in NUM_VARS) {
  boxplot(d[[v]] ~ d$diagnostic, col = "#cce3fd", border = "#1757ac", outcol = "#1757ac",
          pch = 16, cex = 0.5, xlab = "", ylab = labs[[v]], main = labs[[v]], col.main = INK, las = 1)
}
dev.off()

section("3. CATEGORICAL FEATURES")
cat_tab <- do.call(rbind, lapply(names(cat_df), function(v) {
  tab <- table(cat_df[[v]], d$diagnostic)
  share <- prop.table(tab, 2)
  data.frame(variable = v, level = rownames(tab), n_all = as.integer(rowSums(tab)),
             pct_all = round(100 * rowSums(tab) / nrow(d), 1),
             round(100 * unclass(share), 1), row.names = NULL, check.names = FALSE)
}))
print(cat_tab, row.names = FALSE)
save_tab(cat_tab, "categorical_by_class.csv")

# heatmap: rows are feature levels, columns classes, cell = % of class with that level
heat <- function(tab, file, main) {
  M <- as.matrix(tab[CLASSES])
  rn <- paste(tab$variable, tab$level, sep = ": ")
  h <- 400 + 55 * nrow(M)
  png(file.path(OUT_DIR, file), width = 1700, height = h, res = 200)
  par(mar = c(1, 13, 4, 1), fg = GREY)
  image(seq_along(CLASSES), seq_len(nrow(M)), t(M[nrow(M):1, , drop = FALSE]), col = RAMP,
        zlim = c(0, 100), axes = FALSE, xlab = "", ylab = "", main = main, col.main = INK)
  axis(3, at = seq_along(CLASSES), labels = CLASSES, tick = FALSE, col.axis = INK, line = -0.5)
  axis(2, at = seq_len(nrow(M)), labels = rev(rn), las = 1, tick = FALSE, cex.axis = 0.7, col.axis = INK)
  Mr <- M[nrow(M):1, , drop = FALSE]
  text(rep(seq_along(CLASSES), each = nrow(M)), rep(seq_len(nrow(M)), length(CLASSES)),
       sprintf("%.0f", Mr), cex = 0.6, col = ifelse(Mr > 55, "#ffffff", INK))
  abline(h = which(diff(as.integer(factor(rev(tab$variable), levels = unique(rev(tab$variable))))) != 0) + 0.5,
         col = "#ffffff", lwd = 3)
  dev.off()
}
patient_vars <- c("gender", "background_father", "background_mother", LOGICALS)
lesion_vars  <- c("region", "itch", "grew", "hurt", "changed", "bleed", "elevation")
heat(cat_tab[cat_tab$variable %in% patient_vars, ], "patient_features_by_class.png",
     "Patient features: % of each class")
heat(cat_tab[cat_tab$variable %in% lesion_vars, ], "lesion_features_by_class.png",
     "Lesion features: % of each class")

section("4. DOES EACH FEATURE DIFFER BETWEEN CLASSES?")
tests <- rbind(
  do.call(rbind, lapply(NUM_VARS, function(v) {
    kt <- kruskal.test(d[[v]] ~ d$diagnostic)
    data.frame(variable = v, type = "numeric", test = "Kruskal-Wallis",
               statistic = round(unname(kt$statistic), 2), p_value = signif(kt$p.value, 3))
  })),
  do.call(rbind, lapply(names(cat_df), function(v) {
    ct <- chisq.test(table(cat_df[[v]], d$diagnostic), simulate.p.value = TRUE, B = 5000)
    data.frame(variable = v, type = "categorical", test = "Chi-square (simulated p)",
               statistic = round(unname(ct$statistic), 2), p_value = signif(ct$p.value, 3))
  }))
)
tests <- tests[order(tests$p_value, -tests$statistic), ]
print(tests, row.names = FALSE)
save_tab(tests, "feature_tests.csv")

section("SUMMARY")
cat("Files written to", OUT_DIR, ":\n")
print(list.files(OUT_DIR))
