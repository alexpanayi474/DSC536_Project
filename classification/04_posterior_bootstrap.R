# 04_posterior_bootstrap.R: AI-prior classification experiment, the analogue
# of Figure 1 in O'Hagan & Rockova (2025, Sec. 5.1) for PAD-UFES-20.
#
# Input : data/pad_clean.csv                    (01_clean.R)
#         data/llm_classification/llm_probs.csv (03_llm_prior.R)
# Output: classification/results/posterior_bootstrap/
#           splits.rds         the 10 patient splits (labelled, unlabelled, test)
#           results_raw.csv    one row per split x model x alpha (and baselines)
#           results_summary.csv mean and sd over splits
#           macro_f1_vs_alpha.png
#
# Usage, from the project root: source("classification/04_posterior_bootstrap.R")
#   Quick test (2 splits, B = 10, 3 alphas): Sys.setenv(PB_QUICK = "1")
#   Number of cores:               Sys.setenv(PB_CORES = "4")
#
# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------
# Main metric: macro F1 on the test patients (classes are unbalanced, so plain
# accuracy is reported but not used to compare). Balanced accuracy and log
# loss are also reported.
#
# theta is a classifier f_theta(x) giving probabilities for the 6 classes,
# defined as the minimiser of the expected cross entropy under the unknown
# data distribution F (paper, eq. 3.1). Two classifiers are used:
#   LR : multinomial logistic regression (nnet with no hidden layer)
#   MLP: one hidden layer with 20 units, as in the paper's skin example
#
# Prior on F: F ~ DP(alpha, F_AI). Posterior: F | D_n ~ DP(alpha + n, G_n),
# G_n = (alpha F_AI + sum of point masses at the n labelled patients) / (alpha + n).
#
# Construction of F_AI (the AI base measure, paper Sec. 3.2.1):
#   f_AI(Y*, X*) = f_AI^X(X*) * f_AI^Y(Y* | X*)
#   f_AI^X : uniform over the N unlabelled patients of the split, g_j = 1/N.
#            Only their features are used, never their labels.
#   f_AI^Y : the LLM's probabilities for patient j, p*_j1 ... p*_j6, asked once
#            per patient with a prompt fixed before seeing any results.
#   So F_AI puts mass p*_jk / N on the pair (class k, features of patient j):
#   a discrete measure on N x 6 points.
#
# Because F_AI is discrete, a posterior draw of F is a Dirichlet weighting of
# all its points (paper, eq. 3.8):
#   n labelled rows                   Dirichlet parameter 1 each
#   N x 6 AI rows (patient j, class k) Dirichlet parameter alpha * p*_jk / N
# The AI rows together carry parameter alpha, so the prior counts as alpha
# pseudo observations whatever N is, spread over the classes according to the
# LLM's confidence. Rows with p*_jk = 0 have no mass and are dropped.
# This uses the full LLM distribution instead of sampling Y* (paper's galaxy
# example) or keeping only the most likely class (paper's skin example); with
# one hot p* it reduces exactly to the one label per patient version.
#
# Dirichlet weights are drawn as independent Gamma(shape) variables divided by
# their sum. Each draw gives theta(t) by a weighted fit; the posterior
# predictive for a test patient is the average of the B probability vectors,
# and the prediction is the class with the highest average (paper, eq. 5.1).
# alpha = 0 is the Bayesian bootstrap on the labelled data only.
#
# Design: 10 random splits at patient level, stratified by class so the rare
# classes appear in every labelled set; the same splits and the same Dirichlet
# draws are used for every alpha and both classifiers (paired comparison).

library(nnet)
library(parallel)

DATA_DIR  <- "data"
CLEAN     <- file.path(DATA_DIR, "pad_clean.csv")
PROBS     <- file.path(DATA_DIR, "llm_classification", "llm_probs.csv")
OUT_DIR   <- file.path("classification", "results", "posterior_bootstrap")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

QUICK     <- identical(Sys.getenv("PB_QUICK"), "1")
SEED      <- 536
N_SPLITS  <- if (QUICK) 2 else 10
B         <- if (QUICK) 10 else 100
ALPHAS    <- if (QUICK) c(0, 10, 1000) else c(0, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000)
PROP      <- c(lab = 0.15, unl = 0.65, test = 0.20)
HIDDEN    <- 20          # MLP hidden units
DECAY     <- 1e-3        # small weight decay, same for every fit
MAXIT     <- 300
DROP_UNK_LEAK <- FALSE   # TRUE drops grew_UNK and changed_UNK (possible leak, see CLAUDE.md)
N_CORES   <- as.integer(Sys.getenv("PB_CORES", max(1, detectCores() - 1)))

CLASSES   <- c("ACK", "BCC", "SCC", "MEL", "NEV", "SEK")
MODELS    <- c("LR", "MLP")

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
pad   <- read.csv(CLEAN, stringsAsFactors = FALSE)
probs <- read.csv(PROBS, stringsAsFactors = FALSE)
probs <- probs[match(paste(pad$patient_id, pad$lesion_id),
                     paste(probs$patient_id, probs$lesion_id)), ]
stopifnot(!anyNA(probs$p_ACK))

feat_cols <- setdiff(names(pad), c("patient_id", "lesion_id", "diagnostic"))
if (DROP_UNK_LEAK) feat_cols <- setdiff(feat_cols, c("grew_UNK", "changed_UNK"))
num_cols  <- c("age", "fitspatrick", "diameter_1", "diameter_2")

X_all <- as.matrix(pad[feat_cols])
y_all <- match(pad$diagnostic, CLASSES)
P_all <- as.matrix(probs[paste0("p_", CLASSES)])   # LLM probabilities, rows sum to 1
P_all <- P_all / rowSums(P_all)
n_all <- nrow(pad)
onehot <- function(k) { m <- matrix(0, length(k), length(CLASSES)); m[cbind(seq_along(k), k)] <- 1; m }

cat("Patients:", n_all, " features:", length(feat_cols), "\n")
print(table(factor(pad$diagnostic, CLASSES)))

# ---------------------------------------------------------------------------
# 2. Splits (stratified by class, at patient level: one row = one patient)
# ---------------------------------------------------------------------------
make_split <- function(s) {
  set.seed(SEED + s)
  role <- character(n_all)
  for (k in seq_along(CLASSES)) {
    idx <- sample(which(y_all == k))
    n_k <- length(idx)
    n_lab <- max(1, round(PROP["lab"] * n_k)); n_test <- max(1, round(PROP["test"] * n_k))
    role[idx] <- c(rep("lab", n_lab), rep("test", n_test), rep("unl", n_k - n_lab - n_test))
  }
  list(lab = which(role == "lab"), unl = which(role == "unl"), test = which(role == "test"))
}
splits <- lapply(seq_len(N_SPLITS), make_split)
saveRDS(splits, file.path(OUT_DIR, "splits.rds"))
cat("Split sizes (labelled / unlabelled / test):",
    paste(lengths(splits[[1]]), collapse = " / "), "\n")

# ---------------------------------------------------------------------------
# 3. Building blocks
# ---------------------------------------------------------------------------
# numeric features scaled with labelled + unlabelled features (no labels used),
# test rows never contribute to the scaling
scale_split <- function(sp) {
  X <- X_all
  ref <- c(sp$lab, sp$unl)
  for (v in num_cols) {
    m <- mean(X[ref, v]); s <- sd(X[ref, v]); if (s == 0) s <- 1
    X[, v] <- (X[, v] - m) / s
  }
  X
}

# AI rows: each unlabelled patient once per class, with mass p*_jk / N in F_AI
ai_rows <- function(sp, X) {
  N <- length(sp$unl)
  j <- rep(sp$unl, times = length(CLASSES))
  k <- rep(seq_along(CLASSES), each = N)
  p <- as.vector(P_all[sp$unl, ])
  keep <- p > 0
  list(X = X[j[keep], , drop = FALSE], k = k[keep], mass = p[keep] / N)
}

fit_net <- function(X, Y, w, model) {
  if (model == "LR") {
    nnet(X, Y, weights = w, size = 0, skip = TRUE, softmax = TRUE,
         decay = DECAY, maxit = MAXIT, trace = FALSE, MaxNWts = 10000)
  } else {
    nnet(X, Y, weights = w, size = HIDDEN, softmax = TRUE,
         decay = DECAY, maxit = MAXIT, trace = FALSE, MaxNWts = 10000)
  }
}

# Classes are unbalanced (BCC is over half of the patients), so accuracy
# rewards predicting BCC. Main metric: macro F1 (F1 per class, then the plain
# mean, so every class counts the same). Also balanced accuracy (mean recall
# per class), log loss (quality of the probabilities) and plain accuracy.
metrics_from_pred <- function(pred, y, prob = NULL) {
  K  <- length(CLASSES)
  tp <- sapply(1:K, function(k) sum(pred == k & y == k))
  fp <- sapply(1:K, function(k) sum(pred == k & y != k))
  fn <- sapply(1:K, function(k) sum(pred != k & y == k))
  prec <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  rec  <- ifelse(tp + fn > 0, tp / (tp + fn), NA)
  f1   <- ifelse(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
  present <- tp + fn > 0
  ll <- if (is.null(prob)) NA else -mean(log(pmax(prob, 1e-12)[cbind(seq_along(y), y)]))
  c(macro_f1 = mean(f1[present]), balanced_acc = mean(rec[present]),
    accuracy = mean(pred == y), log_loss = ll)
}
metrics <- function(prob, y) metrics_from_pred(max.col(prob, ties.method = "first"), y, prob)

# ---------------------------------------------------------------------------
# 4. One split: posterior bootstrap for every alpha and model, plus baselines
# ---------------------------------------------------------------------------
run_split <- function(s) {
  sp <- splits[[s]]
  X  <- scale_split(sp)
  ai <- ai_rows(sp, X)
  n  <- length(sp$lab)
  X_lab <- X[sp$lab, , drop = FALSE]; Y_lab <- onehot(y_all[sp$lab])
  X_tst <- X[sp$test, , drop = FALSE]; y_tst <- y_all[sp$test]
  X_fit <- rbind(X_lab, ai$X); Y_fit <- rbind(Y_lab, onehot(ai$k))
  out <- list()
  add <- function(model, alpha, m) out[[length(out) + 1]] <<-
    data.frame(split = s, model = model, alpha = alpha, t(m))

  for (a in seq_along(ALPHAS)) {
    alpha <- ALPHAS[a]
    # the same B weight vectors for both classifiers
    set.seed(SEED * 1000 + s * 100 + a)
    W <- replicate(B, {
      g <- c(rgamma(n, shape = 1),
             if (alpha > 0) rgamma(length(ai$mass), shape = alpha * ai$mass) else numeric(length(ai$mass)))
      g / sum(g) * n            # normalised Dirichlet weights, rescaled to sum to n
    })
    for (model in MODELS) {
      pred <- matrix(0, length(y_tst), length(CLASSES))
      for (t in seq_len(B)) {
        w <- W[, t]; use <- w > 1e-10
        fit  <- fit_net(X_fit[use, , drop = FALSE], Y_fit[use, , drop = FALSE], w[use], model)
        pred <- pred + predict(fit, X_tst)
      }
      add(model, alpha, metrics(pred / B, y_tst))
    }
  }

  # baselines
  set.seed(SEED * 1000 + s * 100 + 99)
  for (model in MODELS) {
    # one ordinary fit on the labelled patients only (no prior, no bootstrap)
    fit <- fit_net(X_lab, Y_lab, rep(1, n), model)
    add(paste0(model, "_labelled_only"), NA, metrics(predict(fit, X_tst), y_tst))
    # a fit on the AI rows only, weighted by the LLM probabilities (alpha -> infinity)
    fit <- fit_net(ai$X, onehot(ai$k), ai$mass / sum(ai$mass) * length(sp$unl), model)
    add(paste0(model, "_ai_only"), NA, metrics(predict(fit, X_tst), y_tst))
  }
  add("LLM", NA, metrics(P_all[sp$test, , drop = FALSE], y_tst))
  maj <- which.max(tabulate(y_all[sp$lab], length(CLASSES)))
  add("majority", NA, metrics_from_pred(rep(maj, length(y_tst)), y_tst))

  do.call(rbind, out)
}

# ---------------------------------------------------------------------------
# 5. Run (splits in parallel)
# ---------------------------------------------------------------------------
cat(sprintf("Running %d splits x %d alphas x %d models x B = %d on %d cores\n",
            N_SPLITS, length(ALPHAS), length(MODELS), B, N_CORES))
t0 <- Sys.time()
if (N_CORES > 1) {
  cl <- makeCluster(min(N_CORES, N_SPLITS))
  clusterEvalQ(cl, library(nnet))
  clusterExport(cl, c("splits", "X_all", "y_all", "P_all", "num_cols", "CLASSES", "MODELS",
                      "ALPHAS", "B", "SEED", "HIDDEN", "DECAY", "MAXIT", "onehot",
                      "scale_split", "ai_rows", "fit_net", "metrics", "metrics_from_pred"))
  res <- parLapply(cl, seq_len(N_SPLITS), run_split)
  stopCluster(cl)
} else {
  res <- lapply(seq_len(N_SPLITS), run_split)
}
res <- do.call(rbind, res)
cat(sprintf("Done in %.1f min\n", as.numeric(Sys.time() - t0, units = "mins")))
write.csv(res, file.path(OUT_DIR, "results_raw.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# 6. Summary over splits
# ---------------------------------------------------------------------------
key <- paste(res$model, ifelse(is.na(res$alpha), "", res$alpha))
METRICS <- c("macro_f1", "balanced_acc", "accuracy", "log_loss")
MAIN    <- "macro_f1"
summ <- do.call(rbind, lapply(split(res, key), function(d) {
  m <- unlist(lapply(METRICS, function(v) setNames(c(mean(d[[v]]), sd(d[[v]])),
                                                    paste0(v, c("_mean", "_sd")))))
  data.frame(model = d$model[1], alpha = d$alpha[1], t(m))
}))
summ <- summ[order(summ$model, summ$alpha), ]
rownames(summ) <- NULL
write.csv(summ, file.path(OUT_DIR, "results_summary.csv"), row.names = FALSE)
print(summ, digits = 3)

for (model in MODELS) {
  d <- summ[summ$model == model, ]
  best <- d[which.max(d$macro_f1_mean), ]
  base <- d[d$alpha == 0, ]
  cat(sprintf("%s: best alpha = %g (macro F1 %.3f) vs alpha = 0 (%.3f)\n",
              model, best$alpha, best$macro_f1_mean, base$macro_f1_mean))
}

# ---------------------------------------------------------------------------
# 7. Plot: macro F1 against alpha, as in the paper's Figure 1 (which uses accuracy)
# ---------------------------------------------------------------------------
xpos <- function(a) ifelse(a == 0, log10(min(ALPHAS[ALPHAS > 0])) - 1, log10(a))
png(file.path(OUT_DIR, "macro_f1_vs_alpha.png"), width = 1600, height = 800, res = 130)
par(mfrow = c(1, 2), mar = c(8.5, 4.5, 3, 1))
for (model in MODELS) {
  d  <- summ[summ$model == model, ]
  bl <- summ[is.na(summ$alpha), ]
  mu <- d[[paste0(MAIN, "_mean")]]; sdv <- d[[paste0(MAIN, "_sd")]]
  getb <- function(m) bl[[paste0(MAIN, "_mean")]][bl$model == m]
  ref <- c(getb(paste0(model, "_labelled_only")), getb(paste0(model, "_ai_only")),
           getb("LLM"), getb("majority"))
  x  <- xpos(d$alpha)
  yl <- range(c(mu - sdv, mu + sdv, ref), na.rm = TRUE)
  plot(x, mu, type = "n", ylim = yl, xaxt = "n",
       xlab = expression(alpha), ylab = "Test macro F1",
       main = paste(model, "with AI prior (mean over", N_SPLITS, "splits)"))
  axis(1, at = xpos(ALPHAS), labels = ALPHAS)
  polygon(c(x, rev(x)), c(mu - sdv, rev(mu + sdv)),
          col = adjustcolor("steelblue", 0.2), border = NA)
  lines(x, mu, type = "b", pch = 19, col = "steelblue")
  abline(h = ref, lty = c(2, 3, 4, 5), col = c("black", "darkred", "darkgreen", "grey40"))
  legend("bottom", inset = c(0, -0.42), xpd = TRUE, ncol = 2, bty = "n", cex = 0.8,
         legend = c("Posterior majority vote (+/- 1 sd)", "Labelled only fit", "AI labels only fit",
                    "LLM argmax", "Majority class"),
         lty = c(1, 2, 3, 4, 5), pch = c(19, NA, NA, NA, NA),
         col = c("steelblue", "black", "darkred", "darkgreen", "grey40"))
}
dev.off()
cat("Saved results and plot in", OUT_DIR, "\n")
