# 04_posterior_bootstrap.R: AI-prior classification experiment, the analogue
# of Figure 1 in O'Hagan & Rockova (2025, Sec. 5.1) for PAD-UFES-20.
# Multinomial logistic regression only.
#
# Input : data/pad_clean.csv                    (01_clean.R)
#         data/llm_classification/llm_probs.csv (03_llm_prior.R)
# Output: classification/results/posterior_bootstrap/
#   splits.rds                 the patient splits (labelled, unlabelled, test)
#   draws/split_SS_alpha_A.rds one file per (split, alpha): metrics, posterior
#                              predictive probabilities on the test patients,
#                              and the B coefficient draws (used by 05_inference.R)
#   baselines.csv              labelled only fit, AI only fit, LLM, majority class
#   baselines_summary.csv      mean and sd over splits
#   results_raw.csv            one row per (split, alpha)
#   results_summary.csv        mean and sd over splits for every alpha
#   paired_vs_alpha0.csv       difference to alpha = 0 within each split
#   best_alpha_by_split.csv
#   fig_macro_f1_vs_alpha.png, fig_metrics_vs_alpha.png,
#   fig_per_split.png, fig_per_class_f1.png
#   progress.log
#
# Usage, from the project root: source("classification/04_posterior_bootstrap.R")
#   Quick test (2 splits, B = 20, 4 alphas): Sys.setenv(PB_QUICK = "1")
#   Number of worker processes:              Sys.setenv(PB_CORES = "10")
# Finished (split, alpha) files are skipped, so the script can be stopped and
# started again, and new alphas can be added to ALPHAS without redoing old ones.
# The draws folder is large (coefficients of every draw); keep it out of git.
#
# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------
# theta is a multinomial logistic regression f_theta(x) over the 6 classes,
# defined as the minimiser of the expected cross entropy under the unknown
# data distribution F (paper, eq. 3.1). No model is assumed to be true; theta
# is the best logistic approximation under F.
#
# Prior: F ~ DP(alpha, F_AI). Posterior: F | D_n ~ DP(alpha + n, G_n) with
# G_n = (alpha F_AI + sum of point masses at the n labelled patients) / (alpha + n).
#
# Construction of F_AI (paper, Sec. 3.2.1):
#   f_AI(Y*, X*) = f_AI^X(X*) * f_AI^Y(Y* | X*)
#   f_AI^X : uniform over the N unlabelled patients of the split, g_j = 1/N.
#            Only their features are used, never their labels.
#   f_AI^Y : the LLM's probabilities p*_j1 ... p*_j6 for patient j, asked once
#            per patient with a prompt fixed before any results were seen.
#   F_AI therefore puts mass p*_jk / N on the pair (class k, features of
#   patient j): a discrete measure on N x 6 points.
#
# Because F_AI is discrete, a posterior draw of F is a Dirichlet weighting of
# all its points (paper, eq. 3.8):
#   n labelled rows                    Dirichlet parameter 1 each
#   N x 6 AI rows (patient j, class k) Dirichlet parameter alpha * p*_jk / N
# Together the AI rows carry parameter alpha: the prior is worth alpha pseudo
# observations whatever N is, spread over the classes by the LLM's confidence.
# Rows with p*_jk = 0 have no mass and are dropped. This uses the whole LLM
# distribution instead of sampling Y* (paper's galaxy example) or keeping
# only the most likely class (paper's skin example); with one hot p* it
# reduces exactly to the one label per patient version.
#
# Dirichlet weights = independent Gamma(shape) draws divided by their sum,
# rescaled to sum to n so the small weight decay has the same strength for
# every alpha. Each draw gives theta(t) by a weighted fit; the posterior
# predictive for a test patient is the average of the B probability vectors
# and the prediction is the class with the highest average (paper, eq. 5.1).
# alpha = 0 is the Bayesian bootstrap on the labelled data only.
#
# Speed: every draw starts the optimiser from the fit with the expected
# weights (warm start), rows with negligible weight are left out, and the
# B draws of each (split, alpha) are shared out over parallel workers.
#
# Metrics: classes are unbalanced (BCC is over half of the patients), so the
# main metric is macro F1 (F1 per class, then the plain mean). Macro recall
# and macro precision (also plain means over the classes) show which side of
# F1 moves. Accuracy is reported because the paper uses it, but here it
# mostly rewards predicting BCC: always saying BCC already gives about 0.57.
# Recall, precision and F1 per class and the log loss are saved too.
#
# Design: 10 random splits at patient level (as in the paper), stratified by
# class so the rare classes appear in every labelled set; the same splits for
# every alpha, so differences between alphas are paired within a split.

library(nnet)
library(parallel)

DATA_DIR  <- "data"
CLEAN     <- file.path(DATA_DIR, "pad_clean.csv")
PROBS     <- file.path(DATA_DIR, "llm_classification", "llm_probs.csv")
OUT_DIR   <- file.path("classification", "results", "posterior_bootstrap")
DRAW_DIR  <- file.path(OUT_DIR, "draws")
dir.create(DRAW_DIR, showWarnings = FALSE, recursive = TRUE)
LOG_FILE  <- file.path(OUT_DIR, "progress.log")

QUICK     <- identical(Sys.getenv("PB_QUICK"), "1")
SEED      <- 536
N_SPLITS  <- if (QUICK) 2 else 10
B         <- if (QUICK) 20 else 200
# log grid from 0.1 to 1000, four points per decade, rounded to 3 digits
# so the values are readable in file names and tables
ALPHAS    <- if (QUICK) c(0, 1, 10, 100) else c(0, signif(10^seq(-1, 3, by = 0.25), 3))
PROP      <- c(lab = 0.15, unl = 0.65, test = 0.20)
DECAY     <- 1e-3          # small weight decay, same for every fit
MAXIT     <- 500
RELTOL    <- 1e-6
SAVE_COEF <- TRUE          # keep the coefficient draws for 05_inference.R
N_CORES   <- as.integer(Sys.getenv("PB_CORES", max(1, detectCores() - 1)))

CLASSES   <- c("ACK", "BCC", "SCC", "MEL", "NEV", "SEK")
K         <- length(CLASSES)

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
pad   <- read.csv(CLEAN, stringsAsFactors = FALSE)
probs <- read.csv(PROBS, stringsAsFactors = FALSE)
probs <- probs[match(paste(pad$patient_id, pad$lesion_id),
                     paste(probs$patient_id, probs$lesion_id)), ]
stopifnot(!anyNA(probs$p_ACK))

feat_cols <- setdiff(names(pad), c("patient_id", "lesion_id", "diagnostic"))
num_cols  <- c("age", "fitspatrick", "diameter_1", "diameter_2")
X_all <- as.matrix(pad[feat_cols])
y_all <- match(pad$diagnostic, CLASSES)
P_all <- as.matrix(probs[paste0("p_", CLASSES)])
P_all <- P_all / rowSums(P_all)
n_all <- nrow(pad)

cat("Patients:", n_all, " features:", length(feat_cols), "\n")
print(table(factor(pad$diagnostic, CLASSES)))

# ---------------------------------------------------------------------------
# 2. Splits (stratified by class; one row = one patient)
# ---------------------------------------------------------------------------
make_split <- function(s) {
  set.seed(SEED + s)
  role <- character(n_all)
  for (k in seq_len(K)) {
    idx <- sample(which(y_all == k)); n_k <- length(idx)
    n_lab <- max(1, round(PROP["lab"] * n_k)); n_test <- max(1, round(PROP["test"] * n_k))
    role[idx] <- c(rep("lab", n_lab), rep("test", n_test), rep("unl", n_k - n_lab - n_test))
  }
  list(lab = which(role == "lab"), unl = which(role == "unl"), test = which(role == "test"))
}
splits <- lapply(seq_len(N_SPLITS), make_split)
saveRDS(splits, file.path(OUT_DIR, "splits.rds"))
cat("Split sizes (labelled / unlabelled / test):", paste(lengths(splits[[1]]), collapse = " / "), "\n")

# ---------------------------------------------------------------------------
# 3. Building blocks
# ---------------------------------------------------------------------------
onehot <- function(k) { m <- matrix(0, length(k), K); m[cbind(seq_along(k), k)] <- 1; m }

# numeric features scaled with the labelled and unlabelled features of the
# split (no labels used); the test patients never enter the scaling
prep_split <- function(sp) {
  X <- X_all; sc <- list()
  for (v in num_cols) {
    ref <- X[c(sp$lab, sp$unl), v]; m <- mean(ref); s <- sd(ref); if (s == 0) s <- 1
    X[, v] <- (X[, v] - m) / s; sc[[v]] <- c(mean = m, sd = s)
  }
  # AI rows: each unlabelled patient once per class, mass p*_jk / N in F_AI
  N <- length(sp$unl)
  j <- rep(sp$unl, times = K); k <- rep(seq_len(K), each = N)
  p <- as.vector(P_all[sp$unl, ]); keep <- p > 0
  list(X = X, scale = sc, n = length(sp$lab), N = N,
       X_fit = rbind(X[sp$lab, , drop = FALSE], X[j[keep], , drop = FALSE]),
       Y_fit = rbind(onehot(y_all[sp$lab]), onehot(k[keep])),
       mass  = p[keep] / N,
       X_tst = X[sp$test, , drop = FALSE], y_tst = y_all[sp$test])
}

fit_lr <- function(X, Y, w, init = NULL) {
  args <- list(X, Y, weights = w, size = 0, skip = TRUE, softmax = TRUE, decay = DECAY,
               maxit = MAXIT, reltol = RELTOL, trace = FALSE, MaxNWts = 10000)
  if (!is.null(init)) args$Wts <- init
  do.call(nnet, args)
}

metrics_from_pred <- function(pred, y, prob = NULL) {
  tp <- sapply(1:K, function(k) sum(pred == k & y == k))
  fp <- sapply(1:K, function(k) sum(pred == k & y != k))
  fn <- sapply(1:K, function(k) sum(pred != k & y == k))
  prec <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  rec  <- ifelse(tp + fn > 0, tp / (tp + fn), NA)
  f1   <- ifelse(!is.na(rec) & prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
  ok   <- tp + fn > 0
  ll   <- if (is.null(prob)) NA else -mean(log(pmax(prob, 1e-12)[cbind(seq_along(y), y)]))
  c(macro_f1 = mean(f1[ok]), macro_recall = mean(rec[ok]), macro_precision = mean(prec[ok]),
    accuracy = mean(pred == y), log_loss = ll,
    setNames(f1, paste0("f1_", CLASSES)), setNames(rec, paste0("recall_", CLASSES)),
    setNames(prec, paste0("precision_", CLASSES)))
}
metrics <- function(prob, y) metrics_from_pred(max.col(prob, ties.method = "first"), y, prob)

draw_file <- function(s, alpha) file.path(DRAW_DIR, sprintf("split_%02d_alpha_%s.rds", s, format(alpha, scientific = FALSE)))
task_seed <- function(s, alpha) SEED * 1000 + s * 97 + sum(utf8ToInt(format(alpha, scientific = FALSE)) * seq_len(nchar(format(alpha, scientific = FALSE))))

# ---------------------------------------------------------------------------
# 4. A chunk of posterior draws for one (split, alpha) (runs on a worker)
# ---------------------------------------------------------------------------
# Every draw t has its own seed, so the result does not depend on how the
# draws are shared out between the workers. D (the prepared split) and INIT
# (the warm start for the current alpha) are sent to the workers beforehand.
run_draws <- function(ts, s, alpha) {
  n <- D$n; m <- length(D$mass)
  pred  <- matrix(0, length(D$y_tst), K)
  coefs <- matrix(NA_real_, length(ts), length(INIT))
  for (i in seq_along(ts)) {
    set.seed(task_seed(s, alpha) + ts[i])
    g <- c(rgamma(n, shape = 1), if (alpha > 0) rgamma(m, shape = alpha * D$mass) else numeric(m))
    w <- g / sum(g) * n
    use <- w > 1e-8 * max(w)
    fit <- fit_lr(D$X_fit[use, , drop = FALSE], D$Y_fit[use, , drop = FALSE], w[use], INIT)
    pred <- pred + predict(fit, D$X_tst)
    coefs[i, ] <- fit$wts
  }
  list(pred = pred, coefs = coefs)
}

# ---------------------------------------------------------------------------
# 5. Baselines (cheap, run here)
# ---------------------------------------------------------------------------
cat("\nFitting the baselines on", N_SPLITS, "splits ...\n")
base_rows <- list()
for (s in seq_len(N_SPLITS)) {
  d  <- prep_split(splits[[s]]); n <- d$n
  sp <- splits[[s]]
  f_lab <- fit_lr(d$X_fit[1:n, , drop = FALSE], d$Y_fit[1:n, , drop = FALSE], rep(1, n))
  ai <- (n + 1):nrow(d$X_fit)
  f_ai  <- fit_lr(d$X_fit[ai, , drop = FALSE], d$Y_fit[ai, , drop = FALSE], d$mass / sum(d$mass) * d$N)
  maj   <- which.max(tabulate(y_all[sp$lab], K))
  base_rows[[s]] <- rbind(
    data.frame(split = s, model = "labelled_only_fit", t(metrics(predict(f_lab, d$X_tst), d$y_tst))),
    data.frame(split = s, model = "ai_only_fit",       t(metrics(predict(f_ai, d$X_tst), d$y_tst))),
    data.frame(split = s, model = "llm_argmax",        t(metrics(P_all[sp$test, , drop = FALSE], d$y_tst))),
    data.frame(split = s, model = "majority_class",    t(metrics_from_pred(rep(maj, length(d$y_tst)), d$y_tst))))
}
baselines <- do.call(rbind, base_rows)
write.csv(baselines, file.path(OUT_DIR, "baselines.csv"), row.names = FALSE)

METRICS <- c("macro_f1", "macro_recall", "macro_precision", "accuracy", "log_loss",
             paste0("f1_", CLASSES), paste0("recall_", CLASSES), paste0("precision_", CLASSES))
mean_sd <- function(d) unlist(lapply(METRICS, function(v)
  setNames(c(mean(d[[v]]), sd(d[[v]])), paste0(v, c("_mean", "_sd")))))
base_summ <- do.call(rbind, lapply(split(baselines, baselines$model), function(d)
  data.frame(model = d$model[1], t(mean_sd(d)))))
write.csv(base_summ, file.path(OUT_DIR, "baselines_summary.csv"), row.names = FALSE)
print(base_summ[, c("model", "macro_f1_mean", "macro_recall_mean", "macro_precision_mean", "accuracy_mean")],
      digits = 3, row.names = FALSE)

# ---------------------------------------------------------------------------
# 6. Run the (split, alpha) tasks one after another, the draws in parallel
# ---------------------------------------------------------------------------
tasks <- list()
for (s in seq_len(N_SPLITS)) for (a in ALPHAS) {
  f <- draw_file(s, a)
  done <- file.exists(f) && isTRUE(readRDS(f)$B == B)
  if (!done) tasks[[length(tasks) + 1]] <- list(s = s, alpha = a)
}
n_total <- N_SPLITS * length(ALPHAS)
cat(sprintf("\n%d of %d (split, alpha) tasks already done, %d to run. B = %d, %d workers.\n",
            n_total - length(tasks), n_total, length(tasks), B, N_CORES))
cat("Progress is also written to", LOG_FILE, "\n")

run_task <- function(s, alpha) {
  t0 <- Sys.time()
  # warm start: fit with the expected weights (labelled 1 each, AI alpha * mass)
  n  <- D$n
  w0 <- c(rep(1, n), alpha * D$mass); w0 <- w0 / sum(w0) * n
  use0 <- w0 > 1e-8 * max(w0)
  INIT <<- fit_lr(D$X_fit[use0, , drop = FALSE], D$Y_fit[use0, , drop = FALSE], w0[use0])$wts
  clusterExport(cl, "INIT")

  # draws in chunks; one round = one chunk per worker, progress after each round
  size   <- max(1, ceiling(B / (4 * N_CORES)))
  chunks <- split(seq_len(B), ceiling(seq_len(B) / size))
  rounds <- split(chunks, ceiling(seq_along(chunks) / N_CORES))
  out <- list(); done <- 0
  for (r in rounds) {
    out  <- c(out, parLapplyLB(cl, r, run_draws, s = s, alpha = alpha))
    done <- done + length(unlist(r))
    cat(sprintf("\r    draws %4d / %d  (%.0f s)", done, B, as.numeric(Sys.time() - t0, units = "secs")))
    flush.console()
  }
  cat("\n")

  pred  <- Reduce(`+`, lapply(out, `[[`, "pred")) / B
  coefs <- do.call(rbind, lapply(out, `[[`, "coefs"))
  res <- list(split = s, alpha = alpha, B = B, metrics = metrics(pred, D$y_tst),
              test_idx = splits[[s]]$test, pred = pred,
              coefs = if (SAVE_COEF) signif(coefs, 5) else NULL,
              coef_layout = list(rows = c("(Intercept)", colnames(X_all)), cols = CLASSES),
              scale = D$scale, seconds = as.numeric(Sys.time() - t0, units = "secs"))
  saveRDS(res, draw_file(s, alpha), compress = "xz")
  cat(format(Sys.time(), "%H:%M:%S"), sprintf("split %d alpha %g done in %.0f s, macro F1 %.3f\n",
      s, alpha, res$seconds, res$metrics["macro_f1"]), file = LOG_FILE, append = TRUE)
  res
}

if (length(tasks) > 0) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "start:", length(tasks), "tasks\n", file = LOG_FILE, append = TRUE)
  t_start <- Sys.time()
  cat("Starting", N_CORES, "workers ...\n")
  cl <- makeCluster(N_CORES)
  clusterEvalQ(cl, library(nnet))
  clusterExport(cl, c("K", "fit_lr", "task_seed", "SEED", "DECAY", "MAXIT", "RELTOL"))
  s_loaded <- 0
  for (i in seq_along(tasks)) {
    s <- tasks[[i]]$s; a <- tasks[[i]]$alpha
    if (s != s_loaded) {             # new split: prepare it and send it to the workers
      D <- prep_split(splits[[s]]); clusterExport(cl, "D"); s_loaded <- s
    }
    cat(sprintf("\n[%s] split %d of %d, alpha = %g  (task %d of %d)\n",
                format(Sys.time(), "%H:%M:%S"), s, N_SPLITS, a, i, length(tasks)))
    r <- run_task(s, a)
    elapsed <- as.numeric(Sys.time() - t_start, units = "mins")
    cat(sprintf("    done in %.0f s: macro F1 %.3f, macro recall %.3f, macro precision %.3f, accuracy %.3f\n",
                r$seconds, r$metrics["macro_f1"], r$metrics["macro_recall"],
                r$metrics["macro_precision"], r$metrics["accuracy"]))
    cat(sprintf("    %d / %d tasks done, %.1f min elapsed, about %.1f min left\n",
                i, length(tasks), elapsed, elapsed / i * (length(tasks) - i)))
    flush.console()
  }
  stopCluster(cl)
}

# ---------------------------------------------------------------------------
# 7. Collect results
# ---------------------------------------------------------------------------
cat("\nCollecting results ...\n")
res <- do.call(rbind, lapply(seq_len(N_SPLITS), function(s) do.call(rbind, lapply(ALPHAS, function(a) {
  r <- readRDS(draw_file(s, a)); data.frame(split = s, alpha = a, t(r$metrics), seconds = r$seconds)
}))))
write.csv(res, file.path(OUT_DIR, "results_raw.csv"), row.names = FALSE)

summ <- do.call(rbind, lapply(ALPHAS, function(a) data.frame(alpha = a, t(mean_sd(res[res$alpha == a, ])))))
write.csv(summ, file.path(OUT_DIR, "results_summary.csv"), row.names = FALSE)

# paired comparison with alpha = 0 inside each split (positive macro F1
# difference = AI prior helps; for log loss negative = AI prior helps)
r0 <- res[res$alpha == 0, c("split", "macro_f1", "log_loss")]
paired <- do.call(rbind, lapply(ALPHAS[ALPHAS > 0], function(a) {
  d <- merge(res[res$alpha == a, c("split", "macro_f1", "log_loss")], r0, by = "split", suffixes = c("", "_0"))
  dd <- d$macro_f1 - d$macro_f1_0
  data.frame(alpha = a, macro_f1_diff_mean = mean(dd), macro_f1_diff_sd = sd(dd),
             splits_better = sum(dd > 0), splits = nrow(d),
             paired_t_p = if (nrow(d) > 2 && sd(dd) > 0) t.test(dd)$p.value else NA,
             log_loss_diff_mean = mean(d$log_loss - d$log_loss_0))
}))
write.csv(paired, file.path(OUT_DIR, "paired_vs_alpha0.csv"), row.names = FALSE)

best <- do.call(rbind, lapply(split(res, res$split), function(d)
  data.frame(split = d$split[1], best_alpha = d$alpha[which.max(d$macro_f1)],
             best_macro_f1 = max(d$macro_f1), macro_f1_alpha0 = d$macro_f1[d$alpha == 0])))
write.csv(best, file.path(OUT_DIR, "best_alpha_by_split.csv"), row.names = FALSE)

a_best <- summ$alpha[which.max(summ$macro_f1_mean)]
cat("\n===== Summary over", N_SPLITS, "splits (B =", B, ") =====\n")
print(summ[, c("alpha", "macro_f1_mean", "macro_f1_sd", "macro_recall_mean", "macro_precision_mean", "accuracy_mean")],
      digits = 3, row.names = FALSE)
cat("\nBaselines:\n")
print(base_summ[, c("model", "macro_f1_mean", "macro_f1_sd", "macro_recall_mean", "macro_precision_mean", "accuracy_mean")],
      digits = 3, row.names = FALSE)
cat(sprintf("\nBest alpha by mean macro F1: %g (%.3f) vs alpha = 0 (%.3f)\n",
            a_best, max(summ$macro_f1_mean), summ$macro_f1_mean[summ$alpha == 0]))

# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
cat("Drawing the figures ...\n")
xpos <- function(a) ifelse(a == 0, log10(min(ALPHAS[ALPHAS > 0])) - 1, log10(a))
xaxis <- function() {
  at <- c(xpos(0), log10(10^(floor(log10(min(ALPHAS[ALPHAS > 0]))):ceiling(log10(max(ALPHAS))))))
  axis(1, at = at, labels = c("0", format(10^at[-1], scientific = FALSE, drop0trailing = TRUE)))
}
band_plot <- function(v, ylab, main, baselines_on = TRUE, legend_pos = "bottomleft") {
  mu <- summ[[paste0(v, "_mean")]]; sdv <- summ[[paste0(v, "_sd")]]; x <- xpos(summ$alpha)
  bl <- if (baselines_on) setNames(base_summ[[paste0(v, "_mean")]], base_summ$model) else NULL
  yl <- range(c(mu - sdv, mu + sdv, bl), na.rm = TRUE)
  plot(x, mu, type = "n", ylim = yl, xaxt = "n", xlab = expression(alpha), ylab = ylab, main = main)
  xaxis(); abline(v = (xpos(0) + log10(min(ALPHAS[ALPHAS > 0]))) / 2, col = "grey85")
  polygon(c(x, rev(x)), c(mu - sdv, rev(mu + sdv)), col = adjustcolor("steelblue", 0.2), border = NA)
  lines(x, mu, type = "b", pch = 19, col = "steelblue")
  if (baselines_on) {
    cols <- c(labelled_only_fit = "black", ai_only_fit = "darkred", llm_argmax = "darkgreen", majority_class = "grey40")
    ltys <- c(labelled_only_fit = 2, ai_only_fit = 3, llm_argmax = 4, majority_class = 5)
    for (b in names(cols)) if (!is.na(bl[b])) abline(h = bl[b], col = cols[b], lty = ltys[b])
    legend(legend_pos, bty = "n", cex = 0.75,
           legend = c("AI prior posterior (mean +/- 1 sd)", "Labelled only fit", "AI labels only fit",
                      "LLM argmax", "Majority class"),
           lty = c(1, ltys), pch = c(19, NA, NA, NA, NA), col = c("steelblue", cols))
  }
}

png(file.path(OUT_DIR, "fig_macro_f1_vs_alpha.png"), width = 1300, height = 900, res = 150)
band_plot("macro_f1", "Test macro F1",
          sprintf("Logistic regression with AI prior (%d splits, B = %d)", N_SPLITS, B))
invisible(dev.off())

png(file.path(OUT_DIR, "fig_metrics_vs_alpha.png"), width = 2100, height = 700, res = 150)
par(mfrow = c(1, 3))
band_plot("macro_recall", "Macro recall", "Macro recall", legend_pos = "topleft")
band_plot("macro_precision", "Macro precision", "Macro precision", legend_pos = "topleft")
band_plot("accuracy", "Accuracy", "Accuracy (unbalanced classes)", legend_pos = "bottomleft")
invisible(dev.off())

png(file.path(OUT_DIR, "fig_per_split.png"), width = 1300, height = 900, res = 150)
plot(NA, xlim = range(xpos(ALPHAS)), ylim = range(res$macro_f1), xaxt = "n",
     xlab = expression(alpha), ylab = "Test macro F1", main = "Macro F1 against alpha, one line per split")
xaxis()
for (s in seq_len(N_SPLITS)) { d <- res[res$split == s, ]; lines(xpos(d$alpha), d$macro_f1, col = adjustcolor("grey30", 0.5)) }
lines(xpos(summ$alpha), summ$macro_f1_mean, lwd = 3, col = "steelblue")
points(xpos(best$best_alpha), best$best_macro_f1, pch = 19, col = "darkred")
legend("bottomleft", bty = "n", cex = 0.8, legend = c("One split", "Mean", "Best alpha of a split"),
       lty = c(1, 1, NA), lwd = c(1, 3, NA), pch = c(NA, NA, 19), col = c("grey30", "steelblue", "darkred"))
invisible(dev.off())

png(file.path(OUT_DIR, "fig_per_class_f1.png"), width = 1300, height = 900, res = 150)
cls_col <- c(ACK = "#1b9e77", BCC = "#d95f02", SCC = "#7570b3", MEL = "#e7298a", NEV = "#66a61e", SEK = "#a6761d")
plot(NA, xlim = range(xpos(ALPHAS)), ylim = c(0, 1), xaxt = "n", xlab = expression(alpha),
     ylab = "Test F1 (mean over splits)", main = "F1 per class against alpha")
xaxis()
for (k in CLASSES) lines(xpos(summ$alpha), summ[[paste0("f1_", k, "_mean")]], type = "b", pch = 19, col = cls_col[k])
legend("topleft", bty = "n", legend = CLASSES, col = cls_col, lty = 1, pch = 19, ncol = 3, cex = 0.8)
invisible(dev.off())

cat("\nSaved tables and figures in", OUT_DIR, "\n")
