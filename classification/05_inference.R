# 05_inference.R: posterior inference on the logistic regression coefficients
# for one split and one alpha, with the AI prior against the prior itself and
# against the posterior without AI (alpha = 0, Bayesian bootstrap). 04 runs
# 10 splits to compare alphas; the coefficients are studied on one split only,
# because each split has its own labelled patients and so its own posterior.
#
# Input : classification/results/posterior_bootstrap/ (from 04_posterior_bootstrap.R)
#           splits.rds, results_summary.csv, draws/split_SS_alpha_A.rds
#         data/pad_clean.csv, data/llm_classification/llm_probs.csv
# Output: classification/results/inference/split_SS_alpha_A/
#   coef_summary.csv        every coefficient contrast: prior, posterior, alpha = 0
#   important_predictors.csv features ranked by posterior evidence
#   focus_covariates.csv    the chosen covariates only, with odds ratios
#   fig_forest.png          95% credible intervals, posterior vs alpha = 0
#   fig_density_<feature>.png prior, posterior and alpha = 0 densities per class
#   prior_draws.rds
#
# Usage, from the project root: source("classification/05_inference.R")
#   Choose the split and alpha: Sys.setenv(INF_SPLIT = "1", INF_ALPHA = "316")
#   Default: split 1 and the best alpha by mean macro F1 over the splits
#   (results_summary.csv). INF_ALPHA must be one of the alphas of 04.
#
# ---------------------------------------------------------------------------
# What is estimated
# ---------------------------------------------------------------------------
# Each posterior draw theta(t) of 04_posterior_bootstrap.R is a multinomial
# logistic regression with one coefficient vector per class. With a softmax
# only differences between classes are identified, so every class is compared
# with BCC (the most common diagnosis):
#   beta_kj = theta_kj - theta_BCC,j
# beta_kj is the change in the log odds of class k against BCC when feature j
# goes up by one unit: one standard deviation for age, fitspatrick and the
# diameters (they were scaled), from 0 to 1 for the binary features. For a
# dummy such as region_NOSE the comparison is with the reference level that
# has no column (see 01_clean.R). exp(beta) is the odds ratio.
#
# Prior on theta: F ~ DP(alpha, F_AI) has no labelled data in it, so a prior
# draw of theta is a fit on the AI rows only, with Dirichlet weights
# alpha * p*_jk / N on the (unlabelled patient j, class k) rows. This is the
# prior on theta that the AI prior implies; it is never written down in closed
# form (paper, Sec. 1 and 3.2). Comparing it with the posterior shows how much
# the 15% labelled patients moved the AI's beliefs.
#
# These are posterior summaries of theta(F), the best logistic approximation
# under F; they describe associations in this clinic population, not causes.

library(nnet)
library(parallel)

PB_DIR   <- file.path("classification", "results", "posterior_bootstrap")
CLEAN    <- file.path("data", "pad_clean.csv")
PROBS    <- file.path("data", "llm_classification", "llm_probs.csv")
CLASSES  <- c("ACK", "BCC", "SCC", "MEL", "NEV", "SEK")
K        <- length(CLASSES)
REF      <- "BCC"
B_PRIOR  <- 1000
DECAY    <- 1e-3; MAXIT <- 500; RELTOL <- 1e-6      # same as 04
N_CORES  <- as.integer(Sys.getenv("PB_CORES", max(1, parallel::detectCores() - 1)))
FOCUS    <- c("age", "fitspatrick", "diameter_1", "skin_cancer_history", "cancer_history",
              "smoke", "drink", "pesticide", "gender_MALE", "bleed_True", "grew_False",
              "hurt_True", "itch_False", "elevation_False", "region_NOSE", "region_BACK")

summ  <- read.csv(file.path(PB_DIR, "results_summary.csv"))
SPLIT <- as.integer(Sys.getenv("INF_SPLIT", "1"))
ALPHA <- as.numeric(Sys.getenv("INF_ALPHA", summ$alpha[which.max(summ$macro_f1_mean)]))
if (ALPHA == 0) stop("Choose alpha > 0: alpha = 0 is the comparison posterior.")

draw_file <- function(s, a) file.path(PB_DIR, "draws", sprintf("split_%02d_alpha_%s.rds", s, format(a, scientific = FALSE)))
stopifnot(file.exists(draw_file(SPLIT, ALPHA)), file.exists(draw_file(SPLIT, 0)))
post  <- readRDS(draw_file(SPLIT, ALPHA))
post0 <- readRDS(draw_file(SPLIT, 0))
OUT   <- file.path("classification", "results", "inference",
                   sprintf("split_%02d_alpha_%s", SPLIT, format(ALPHA, scientific = FALSE)))
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
cat(sprintf("Inference for split %d, alpha = %g (B = %d posterior draws)\n", SPLIT, ALPHA, post$B))

# ---------------------------------------------------------------------------
# 1. Prior draws of theta: fits on the AI rows with DP(alpha, F_AI) weights
# ---------------------------------------------------------------------------
pad   <- read.csv(CLEAN, stringsAsFactors = FALSE)
probs <- read.csv(PROBS, stringsAsFactors = FALSE)
probs <- probs[match(paste(pad$patient_id, pad$lesion_id), paste(probs$patient_id, probs$lesion_id)), ]
feat  <- post$coef_layout$rows[-1]
X     <- as.matrix(pad[feat])
for (v in names(post$scale)) X[, v] <- (X[, v] - post$scale[[v]]["mean"]) / post$scale[[v]]["sd"]
P     <- as.matrix(probs[paste0("p_", CLASSES)]); P <- P / rowSums(P)
sp    <- readRDS(file.path(PB_DIR, "splits.rds"))[[SPLIT]]
N     <- length(sp$unl); n <- length(sp$lab)
j     <- rep(sp$unl, times = K); k <- rep(seq_len(K), each = N)
p     <- as.vector(P[sp$unl, ]); keep <- p > 0
X_ai  <- X[j[keep], , drop = FALSE]
Y_ai  <- matrix(0, sum(keep), K); Y_ai[cbind(seq_len(sum(keep)), k[keep])] <- 1
mass  <- p[keep] / N

fit_lr <- function(X, Y, w, init = NULL) {
  args <- list(X, Y, weights = w, size = 0, skip = TRUE, softmax = TRUE, decay = DECAY,
               maxit = MAXIT, reltol = RELTOL, trace = FALSE, MaxNWts = 10000)
  if (!is.null(init)) args$Wts <- init
  do.call(nnet, args)$wts
}

f_prior <- file.path(OUT, "prior_draws.rds")
if (file.exists(f_prior)) {
  cat("Reading the saved prior draws from", f_prior, "\n")
  prior <- readRDS(f_prior)
} else {
  cat("Drawing", B_PRIOR, "prior draws of theta ...\n")
  init <- fit_lr(X_ai, Y_ai, mass / sum(mass) * n)
  one_prior <- function(t) {
    set.seed(10000 + t)
    g <- rgamma(length(mass), shape = ALPHA * mass); w <- g / sum(g) * n
    use <- w > 1e-8 * max(w)
    fit_lr(X_ai[use, , drop = FALSE], Y_ai[use, , drop = FALSE], w[use], init)
  }
  cl <- makeCluster(N_CORES)
  clusterEvalQ(cl, library(nnet))
  clusterExport(cl, c("X_ai", "Y_ai", "mass", "n", "ALPHA", "init", "fit_lr", "DECAY", "MAXIT", "RELTOL"))
  # in rounds so the progress can be printed
  rounds <- split(seq_len(B_PRIOR), ceiling(seq_len(B_PRIOR) / (10 * N_CORES)))
  prior <- NULL; t0 <- Sys.time()
  for (r in rounds) {
    prior <- rbind(prior, do.call(rbind, parLapply(cl, r, one_prior)))
    cat(sprintf("\r    prior draws %4d / %d  (%.0f s)", nrow(prior), B_PRIOR,
                as.numeric(Sys.time() - t0, units = "secs")))
    flush.console()
  }
  cat("\n")
  prior <- signif(prior, 5)
  stopCluster(cl)
  saveRDS(prior, f_prior, compress = "xz")
}

# ---------------------------------------------------------------------------
# 2. Coefficient contrasts against BCC
# ---------------------------------------------------------------------------
# a draw is stored as one vector: for each class, intercept then features
cat("Summarising the coefficient draws ...\n")
rows <- post$coef_layout$rows; p1 <- length(rows)
contrasts <- function(W) {
  r <- match(REF, CLASSES)
  out <- lapply(setdiff(seq_len(K), r), function(kk) {
    m <- W[, (kk - 1) * p1 + seq_len(p1), drop = FALSE] - W[, (r - 1) * p1 + seq_len(p1), drop = FALSE]
    colnames(m) <- paste0(CLASSES[kk], ":", rows); m
  })
  do.call(cbind, out)
}
C_post  <- contrasts(post$coefs)
C_post0 <- contrasts(post0$coefs)
C_prior <- contrasts(prior)

summ_draws <- function(M, tag) {
  data.frame(term = colnames(M),
             mean = colMeans(M), sd = apply(M, 2, sd),
             q2.5 = apply(M, 2, quantile, 0.025), q97.5 = apply(M, 2, quantile, 0.975),
             p_pos = colMeans(M > 0), row.names = NULL, check.names = FALSE) |>
    setNames(c("term", paste0(tag, c("_mean", "_sd", "_q2.5", "_q97.5", "_p_pos"))))
}
tab <- Reduce(function(a, b) merge(a, b, by = "term", sort = FALSE),
              list(summ_draws(C_post, "post"), summ_draws(C_post0, "alpha0"), summ_draws(C_prior, "prior")))
tab$class   <- sub(":.*", "", tab$term)
tab$feature <- sub("^[^:]*:", "", tab$term)
tab$post_OR <- exp(tab$post_mean)
tab$post_excludes_0  <- tab$post_q2.5 > 0 | tab$post_q97.5 < 0
tab$sd_ratio_to_alpha0 <- tab$post_sd / tab$alpha0_sd
tab <- tab[, c("class", "feature", setdiff(names(tab), c("class", "feature", "term")))]
write.csv(tab, file.path(OUT, "coef_summary.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# 3. Important predictors: strength of posterior evidence per feature
# ---------------------------------------------------------------------------
# for each feature: the largest |posterior mean / sd| over the five contrasts,
# and in how many contrasts the 95% credible interval excludes 0
ft  <- tab[tab$feature != "(Intercept)", ]
imp <- do.call(rbind, lapply(split(ft, ft$feature), function(d) {
  z <- abs(d$post_mean / d$post_sd); i <- which.max(z)
  data.frame(feature = d$feature[1], max_abs_z = max(z), strongest_contrast = paste(d$class[i], "vs", REF),
             mean_at_strongest = d$post_mean[i], classes_ci_excludes_0 = sum(d$post_excludes_0),
             max_abs_z_alpha0 = max(abs(d$alpha0_mean / d$alpha0_sd)))
}))
imp <- imp[order(-imp$max_abs_z), ]
write.csv(imp, file.path(OUT, "important_predictors.csv"), row.names = FALSE)

# focus covariates with odds ratios; numeric features also per original unit
foc <- tab[tab$feature %in% FOCUS, c("class", "feature", "post_mean", "post_q2.5", "post_q97.5",
                                     "post_p_pos", "alpha0_mean", "alpha0_q2.5", "alpha0_q97.5",
                                     "prior_mean", "prior_q2.5", "prior_q97.5")]
foc$post_OR <- exp(foc$post_mean); foc$post_OR_q2.5 <- exp(foc$post_q2.5); foc$post_OR_q97.5 <- exp(foc$post_q97.5)
foc$unit <- ifelse(foc$feature %in% names(post$scale),
                   paste("1 sd =", round(sapply(foc$feature, function(v) if (v %in% names(post$scale)) post$scale[[v]]["sd"] else NA), 1)),
                   "0 to 1")
foc <- foc[order(match(foc$feature, FOCUS), match(foc$class, CLASSES)), ]
write.csv(foc, file.path(OUT, "focus_covariates.csv"), row.names = FALSE)

cat("\nTop 10 predictors by posterior evidence:\n")
print(head(imp, 10), digits = 3, row.names = FALSE)

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
cat("\nDrawing the figures ...\n")
others <- setdiff(CLASSES, REF)
col3   <- c(prior = "grey55", alpha0 = "darkorange", post = "steelblue")

# forest plot: focus covariates, one panel per class contrast
png(file.path(OUT, "fig_forest.png"), width = 2400, height = 1400, res = 150)
par(mfrow = c(1, length(others)), mar = c(4.5, 1, 3, 0.5), oma = c(0, 9, 3, 0))
yy <- rev(seq_along(FOCUS))
for (cl in others) {
  d  <- tab[tab$class == cl, ]; d <- d[match(FOCUS, d$feature), ]
  xl <- range(c(d$post_q2.5, d$post_q97.5, d$alpha0_q2.5, d$alpha0_q97.5), na.rm = TRUE)
  plot(NA, xlim = xl, ylim = c(0.5, length(FOCUS) + 0.5), yaxt = "n",
       xlab = "log odds vs BCC", ylab = "", main = paste(cl, "vs", REF))
  abline(v = 0, col = "grey60"); abline(h = yy, col = "grey93")
  if (cl == others[1]) axis(2, at = yy, labels = FOCUS, las = 1, cex.axis = 0.85, outer = FALSE, line = 0)
  segments(d$alpha0_q2.5, yy - 0.15, d$alpha0_q97.5, yy - 0.15, col = col3["alpha0"], lwd = 2)
  points(d$alpha0_mean, yy - 0.15, pch = 19, col = col3["alpha0"], cex = 0.8)
  segments(d$post_q2.5, yy + 0.15, d$post_q97.5, yy + 0.15, col = col3["post"], lwd = 2)
  points(d$post_mean, yy + 0.15, pch = 19, col = col3["post"], cex = 0.8)
}
mtext(sprintf("Posterior means and 95%% credible intervals, split %d: AI prior alpha = %g (blue) vs alpha = 0 (orange)",
              SPLIT, ALPHA), outer = TRUE, cex = 0.9)
dev.off()

# densities: prior, posterior with AI prior, posterior with alpha = 0
for (v in FOCUS) {
  png(file.path(OUT, sprintf("fig_density_%s.png", v)), width = 2400, height = 550, res = 150)
  par(mfrow = c(1, length(others)), mar = c(4.5, 4, 3, 0.5), oma = c(0, 0, 2, 0))
  for (cl in others) {
    term <- paste0(cl, ":", v)
    dl <- list(prior = density(C_prior[, term]), alpha0 = density(C_post0[, term]), post = density(C_post[, term]))
    xl <- quantile(c(C_prior[, term], C_post0[, term], C_post[, term]), c(0.005, 0.995))
    yl <- c(0, max(sapply(dl, function(d) max(d$y))))
    plot(NA, xlim = xl, ylim = yl, xlab = "log odds vs BCC", ylab = "Density", main = paste(cl, "vs", REF))
    abline(v = 0, col = "grey70")
    for (nm in names(dl)) lines(dl[[nm]], col = col3[nm], lwd = 2, lty = if (nm == "prior") 2 else 1)
    if (cl == others[1]) legend("topleft", bty = "n", cex = 0.75, lwd = 2, lty = c(2, 1, 1), col = col3,
                                legend = c(paste0("Prior (alpha = ", ALPHA, ")"), "Posterior, alpha = 0",
                                           paste0("Posterior, alpha = ", ALPHA)))
  }
  mtext(paste("Coefficient of", v), outer = TRUE, cex = 1)
  dev.off()
}

cat("\nSaved tables and figures in", OUT, "\n")
