# ==========================================================================
# regression_power_prior.R -- AI-prior Bayesian linear regression for ai_hours_per_week,
# following O'Hagan & Rockova (2025), "AI-Powered Bayesian Inference" (arXiv:2502.19231).
#
# alpha = prior strength = effective number of AI ("fake") data points (paper, Sec. 4.1).
# Two AI priors are compared, both using the LLM's full predictive distribution per student:
#
#  [power]  Power / catalytic AI prior (paper Sec. 2.1.1, eq. 2.5), conjugate Normal-Inverse-Gamma:
#           p(beta, s2 | data) ∝ L(known) * exp( (alpha/m) * sum_AI E_LLM[log L(AI row)] ) * p0
#           Closed form; E_LLM needs only E[y] and E[y^2] of the LLM bin distribution (exact).
#  [pb]     Paper's main method: Dirichlet-process prior DP(alpha, F_AI) sampled with the
#           Posterior Bootstrap (paper Sec. 3.2, Algorithm 1). Each draw: m fake rows with
#           features resampled from the AI rows and labels sampled from the LLM distribution,
#           Dirichlet(1,...,1, alpha/m,...,alpha/m) weights, weighted least squares.
#
# Choosing alpha (paper Sec. 4.1):
#  [cv]     out-of-sample: 10-fold CV log predictive density on the known rows (power prior)
#  [cal]    bootstrap coverage calibration (Sec. 4.1.1, after Syring & Martin): resample the known
#           rows, take the known-data estimate as the bootstrap-world truth, choose the LARGEST
#           alpha whose 90% interval still covers it >= 90% of the time. For pb the interval inside
#           this loop uses the paper's Theorem 1 normal approximation (cheap); final pb intervals
#           are the 5%/95% quantiles of real posterior-bootstrap draws.
#
# Analyses:  A. the saved split (300/100 known rows, all 900 LLM rows)
#            B. N_REP random splits (N_AI_REP LLM rows, known rows from the other real rows,
#               remaining rows = unseen test set). Truth = OLS on all rows not used as known rows.
#
# Inputs: llm_split_v2.rds, llm_probs_v2.rds, campus_ai_pulse_2026.csv. Outputs: regression_output/.
# Base R only. bapr2.R and the data files are not modified.
# Run from the project root: source("regression/regression_power_prior.R")  (~20 min)
# Quick test: Sys.setenv(N_REP = "2", B_PB = "50", B_CAL = "50") before source().
# ==========================================================================

DATA_DIR <- Sys.getenv("CAMPUS_DATA_DIR", "data")                  # csv
LLM_DIR  <- Sys.getenv("LLM_DIR", file.path(DATA_DIR, "llm"))      # llm_split_v2.rds, llm_probs_v2.rds
TAG      <- Sys.getenv("LLM_TAG", "v2")                            # "v3" after llm_prior_labels_v3.R
# How the AI rows enter the prior:
#   "sample" (default): labels drawn from the LLM's full predictive distribution (paper Sec. 3.2.1)
#   "point"           : ONE prediction per student = mean of the LLM's distribution, as in the paper's
#                       regression example (Sec. 5.4), where F_AI is the empirical distribution of
#                       single AI predictions. Because OLS is linear in y, both give the same expected
#                       estimate; "point" drops the LLM's stated uncertainty (narrower intervals).
AI_LABELS <- Sys.getenv("AI_LABELS", "sample")
stopifnot(AI_LABELS %in% c("sample", "point"))
OUT_BASE <- file.path("regression", if (TAG == "v2") "output" else paste0("output_", TAG))
OUT_DIR  <- Sys.getenv("REG_OUT_DIR", if (AI_LABELS == "point") paste0(OUT_BASE, "_point") else OUT_BASE)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

SEED        <- 30175
LEVEL       <- 0.90                                   # credible level, as in the paper
ALPHA_GRID  <- c(0, 1, 3, 10, 30, 100, 300, 1000)    # n_AI (= 1:1 pooling) is added automatically
N_KNOWN_SET <- c(300, 100)
K_FOLDS     <- 10
N_REP       <- as.integer(Sys.getenv("N_REP", "100"))
N_AI_REP    <- 600
B_PB        <- as.integer(Sys.getenv("B_PB", "200"))  # posterior-bootstrap draws per alpha
B_CAL       <- as.integer(Sys.getenv("B_CAL", "200")) # bootstrap resamples for calibration
X_KEY       <- "ai_hours_per_week"
OUTCOMES    <- c("critical_thinking_score", "cgpa")

DROP_COLS <- c("uses_ai", "paid_ai_subscription", "prompt_skill_level", "time_saved_pct",
               "ai_dependency_score", "ai_dependency_level", "integrity_concern_level",
               "flagged_by_ai_detector", "would_recommend_ai")      # same as bapr2.R, by name
# primary_use_case "No AI use" == primary_ai_tool "No AI tool" (same 148 rows, aliased).
# Add "ai_assignment_share_pct" to also remove its r = 0.87 collinearity with ai_hours.
EXCLUDE_ALL <- c("primary_use_case")

PRIOR_PREC <- 1e-6; A0 <- 0.001; B0 <- 0.001                       # vague base prior
Q_T <- function(df) qt(1 - (1 - LEVEL) / 2, df)
Z   <- qnorm(1 - (1 - LEVEL) / 2)

# --------------------------------------------------------------------------
# 1. Data, design matrices, LLM distributions
# --------------------------------------------------------------------------
data <- read.csv(file.path(DATA_DIR, "campus_ai_pulse_2026.csv"), stringsAsFactors = TRUE)
data <- data[, setdiff(names(data), DROP_COLS)]
data$student_id <- as.character(data$student_id)
N <- nrow(data)

split <- readRDS(file.path(LLM_DIR, paste0("llm_split_", TAG, ".rds")))
probs <- readRDS(file.path(LLM_DIR, paste0("llm_probs_", TAG, ".rds")))
stopifnot(identical(probs$student_id, split$ai_ids), !anyNA(probs$exp_cgpa), !anyNA(probs$exp_score))
llm_rows <- match(split$ai_ids, data$student_id)
# with fewer LLM rows (v3 is limited by the API quota) draw ~85% of them per replicate, so the
# AI part still varies between splits; v2 (900 rows) keeps 600
N_AI_REP <- min(N_AI_REP, floor(0.85 * length(llm_rows)))
cat("LLM tag", TAG, ":", length(llm_rows), "AI rows available,", N_AI_REP, "used per random split\n")

make_llm <- function(cols, lo, hi) {
  P <- matrix(NA_real_, N, length(lo)); P[llm_rows, ] <- as.matrix(probs[cols])
  list(P = P, cum = t(apply(P, 1, cumsum)), lo = lo, hi = hi,
       m1 = drop(P %*% ((lo + hi) / 2)), m2 = drop(P %*% ((lo^2 + lo * hi + hi^2) / 3)))
}
LLM <- list(cgpa = make_llm(paste0("p_cgpa_", 1:8), seq(0, 3.5, .5), seq(.5, 4, .5)),
            critical_thinking_score = make_llm(paste0("p_score_", 1:10), seq(0, 90, 10), seq(10, 100, 10)))
draw_llm <- function(L, idx) {                      # one label per row from the LLM distribution
  k <- pmin(1 + rowSums(L$cum[idx, , drop = FALSE] < runif(length(idx))), length(L$lo))
  runif(length(idx), L$lo[k], L$hi[k])
}

if (AI_LABELS == "point") {                         # paper's regression set-up: one prediction per case
  for (y in names(LLM)) LLM[[y]]$m2 <- LLM[[y]]$m1^2  # no within-student spread
  draw_llm <- function(L, idx) L$m1[idx]             # posterior bootstrap uses the point prediction
}
cat("AI labels used in the prior:", if (AI_LABELS == "point") "one point prediction per student (paper Sec. 5.4)"
    else "drawn from the LLM's full distribution", "\n")

# Point predictions (mean of the LLM distribution) for every student that has one, for the PPI
# comparison: AI rows from llm_probs_<tag>.rds, labelled rows from llm_probs_known_<tag>.rds.
f_known <- file.path(LLM_DIR, paste0("llm_probs_known_", TAG, ".rds"))
HAS_PPI <- file.exists(f_known)
PRED <- list(cgpa = LLM$cgpa$m1, critical_thinking_score = LLM$critical_thinking_score$m1)
if (HAS_PPI) {
  pk <- readRDS(f_known); kr <- match(pk$student_id, data$student_id)
  stopifnot(!anyNA(kr), !anyNA(pk$exp_cgpa), !anyNA(pk$exp_score))
  PRED$cgpa[kr] <- pk$exp_cgpa; PRED$critical_thinking_score[kr] <- pk$exp_score
}
cat("PPI comparison:", if (HAS_PPI) paste("yes,", sum(!is.na(PRED$cgpa)), "students with LLM predictions")
    else "no (no LLM predictions for the labelled students)", "\n")

covars_all <- setdiff(names(data), c("student_id", OUTCOMES, EXCLUDE_ALL))
models <- list(score_only = list(y = "critical_thinking_score", x = X_KEY),
               score_all  = list(y = "critical_thinking_score", x = covars_all),
               cgpa_only  = list(y = "cgpa", x = X_KEY),
               cgpa_all   = list(y = "cgpa", x = covars_all))
for (mn in names(models)) {
  models[[mn]]$X <- model.matrix(reformulate(models[[mn]]$x), data)
  models[[mn]]$j <- which(colnames(models[[mn]]$X) == X_KEY)
}

# --------------------------------------------------------------------------
# 2. Power prior (conjugate NIG on sufficient statistics)
# --------------------------------------------------------------------------
ss_xy  <- function(X, y) list(XtX = crossprod(X), Xty = crossprod(X, y), yty = sum(y^2), n = nrow(X))
ss_llm <- function(X, L, idx) list(XtX = crossprod(X[idx, , drop = FALSE]),
                                   Xty = crossprod(X[idx, , drop = FALSE], L$m1[idx]),
                                   yty = sum(L$m2[idx]), n = length(idx))
combine <- function(k, a, w) list(XtX = k$XtX + w * a$XtX, Xty = k$Xty + w * a$Xty,
                                  yty = k$yty + w * a$yty, n = k$n + w * a$n)
post_ss <- function(s) {
  P <- s$XtX; diag(P) <- diag(P) + PRIOR_PREC
  V <- solve(P); m <- drop(V %*% s$Xty)
  rss <- s$yty - 2 * sum(m * s$Xty) + drop(t(m) %*% s$XtX %*% m)
  list(m = m, V = V, a = A0 + s$n / 2, b = B0 + 0.5 * (max(rss, 0) + PRIOR_PREC * sum(m^2)))
}
pp_ci <- function(post, j) {
  s <- sqrt(post$b / post$a * post$V[j, j]); q <- Q_T(2 * post$a)
  c(mean = post$m[j], lo = post$m[j] - q * s, hi = post$m[j] + q * s)
}
log_pred <- function(post, X, y) {
  mu <- drop(X %*% post$m)
  s  <- sqrt(post$b / post$a * (1 + rowSums((X %*% post$V) * X)))
  sum(dt((y - mu) / s, df = 2 * post$a, log = TRUE) - log(s))
}

# --------------------------------------------------------------------------
# 3. Posterior bootstrap (paper Algorithm 1) and its Theorem-1 normal approximation
# --------------------------------------------------------------------------
pb_draws <- function(Xk, yk, X, L, ai_idx, alpha, j, B) {
  n <- nrow(Xk); m <- length(ai_idx)
  vapply(seq_len(B), function(t) {
    wk <- rgamma(n, 1)
    G <- crossprod(Xk, Xk * wk); h <- crossprod(Xk, yk * wk); tot <- sum(wk)
    if (alpha > 0) {
      s  <- ai_idx[sample.int(m, m, replace = TRUE)]       # X* ~ empirical AI features
      Xs <- X[s, , drop = FALSE]; ys <- draw_llm(L, s)     # Y* ~ LLM distribution
      ws <- rgamma(m, alpha / m)                           # Dirichlet(alpha/m) weights
      G <- G + crossprod(Xs, Xs * ws); h <- h + crossprod(Xs, ys * ws); tot <- tot + sum(ws)
    }
    diag(G) <- diag(G) + PRIOR_PREC * tot
    solve(G, h)[j]
  }, 0)
}
# Theorem 1: theta* ≈ N(theta_hat, J^-1 I J^-1 / (n (1 + gamma))), gamma = alpha / n,
# with the AI part integrated over the LLM distribution (E[y*], E[y*^2]).
pb_asym <- function(Xk, yk, aiX, alpha, j) {
  n <- nrow(Xk); g <- alpha / n
  A <- crossprod(Xk) / n + g * aiX$XX; diag(A) <- diag(A) + PRIOR_PREC
  th <- solve(A, crossprod(Xk, yk) / n + g * aiX$Xm1)
  rk <- drop(yk - Xk %*% th)
  Ib <- crossprod(Xk * rk) / n
  if (g > 0) {
    f  <- drop(aiX$X %*% th)
    Ib <- Ib + g * crossprod(aiX$X, aiX$X * pmax(aiX$m2 - 2 * aiX$m1 * f + f^2, 0)) / nrow(aiX$X)
  }
  Ai <- solve(A); S <- Ai %*% Ib %*% Ai / n           # = J^-1 I J^-1 / (n(1+g))
  se <- sqrt(S[j, j])
  c(mean = th[j], lo = th[j] - Z * se, hi = th[j] + Z * se)
}

# --------------------------------------------------------------------------
# 3b. Benchmark used by the paper (Sec. 5.4): prediction-powered inference for an OLS coefficient
#     (Angelopoulos et al. 2023) and its power-tuned version PPI++ (Angelopoulos, Duchi & Zrnic 2023).
#     theta(lambda) = theta_Y[labelled] + lambda * (theta_f[unlabelled] - theta_f[labelled]),
#     where theta_f is OLS on the LLM point predictions. lambda = 1: PPI (the "rectifier"
#     theta_f[labelled] - theta_Y[labelled] removes the LLM's bias); lambda = 0: real data only;
#     PPI++ chooses the lambda that minimises the variance. Sandwich (robust) standard errors.
# --------------------------------------------------------------------------
ols_if <- function(X, y, j) {                        # coefficient j and its influence values
  H  <- crossprod(X) / nrow(X); diag(H) <- diag(H) + PRIOR_PREC
  Hi <- solve(H); b <- drop(Hi %*% (crossprod(X, y) / nrow(X)))
  list(b = unname(b[j]), psi = drop(X %*% Hi[j, ]) * drop(y - X %*% b))
}
ppi <- function(Xk, yk, fk, Xu, fu, j) {
  n <- nrow(Xk); M <- nrow(Xu)
  oy <- ols_if(Xk, yk, j); ofl <- ols_if(Xk, fk, j); ofu <- ols_if(Xu, fu, j)
  vyy <- var(oy$psi) / n; vfl <- var(ofl$psi) / n; vfu <- var(ofu$psi) / M
  cyf <- cov(oy$psi, ofl$psi) / n
  est <- function(lam) {
    m  <- oy$b + lam * (ofu$b - ofl$b)
    se <- sqrt(max(vyy - 2 * lam * cyf + lam^2 * (vfl + vfu), 0))
    c(mean = m, lo = m - Z * se, hi = m + Z * se, lambda = lam)
  }
  list(ppi = est(1), ppi_pp = est(min(max(cyf / (vfl + vfu), 0), 1)))
}

# --------------------------------------------------------------------------
# 4. One analysis (all models x all alphas) for given known / AI / test rows
# --------------------------------------------------------------------------
analyse <- function(known_idx, ai_idx, test_idx, rep_id) {
  n_ai   <- length(ai_idx)
  alphas <- sort(unique(c(ALPHA_GRID, n_ai)))
  nk     <- length(known_idx)
  fold   <- sample(rep(seq_len(K_FOLDS), length.out = nk))
  boots  <- replicate(B_CAL, sample.int(nk, nk, replace = TRUE), simplify = FALSE)
  truth_idx <- setdiff(seq_len(N), known_idx)
  out <- list()
  for (mi in seq_along(models)) {
    md <- models[[mi]]; X <- md$X; y <- data[[md$y]]; L <- LLM[[md$y]]; j <- md$j
    Xk <- X[known_idx, , drop = FALSE]; yk <- y[known_idx]
    tf <- lm.fit(X[truth_idx, , drop = FALSE], y[truth_idx]); truth <- tf$coefficients[[j]]
    sa  <- ss_llm(X, L, ai_idx); sk <- ss_xy(Xk, yk)
    Xa  <- X[ai_idx, , drop = FALSE]
    aiX <- list(X = Xa, m1 = L$m1[ai_idx], m2 = L$m2[ai_idx],
                XX = crossprod(Xa) / n_ai, Xm1 = crossprod(Xa, L$m1[ai_idx]) / n_ai)
    folds <- lapply(seq_len(K_FOLDS), function(k)
      list(ss = ss_xy(Xk[fold != k, , drop = FALSE], yk[fold != k]), te = which(fold == k)))

    # bootstrap coverage calibration (paper Sec. 4.1.1); bootstrap-world truth = known-data estimate
    ref <- post_ss(sk)$m[j]
    cov_pp <- cov_pb <- numeric(length(alphas))
    for (bi in boots) {
      Xb <- Xk[bi, , drop = FALSE]; yb <- yk[bi]; sb <- ss_xy(Xb, yb)
      for (a in seq_along(alphas)) {
        c1 <- pp_ci(post_ss(combine(sb, sa, alphas[a] / n_ai)), j)
        c2 <- pb_asym(Xb, yb, aiX, alphas[a], j)
        cov_pp[a] <- cov_pp[a] + (c1[2] <= ref && ref <= c1[3])
        cov_pb[a] <- cov_pb[a] + (c2[2] <= ref && ref <= c2[3])
      }
    }
    cov_pp <- cov_pp / B_CAL; cov_pb <- cov_pb / B_CAL

    for (a in seq_along(alphas)) {
      al <- alphas[a]; w <- al / n_ai
      post <- post_ss(combine(sk, sa, w)); ci <- pp_ci(post, j)
      cv <- sum(vapply(folds, function(f) log_pred(post_ss(combine(f$ss, sa, w)),
                                                     Xk[f$te, , drop = FALSE], yk[f$te]), 0)) / nk
      te_lpd <- te_rmse <- NA_real_
      if (length(test_idx)) {
        Xt <- X[test_idx, , drop = FALSE]
        te_lpd  <- log_pred(post, Xt, y[test_idx]) / length(test_idx)
        te_rmse <- sqrt(mean((y[test_idx] - Xt %*% post$m)^2))
      }
      d  <- pb_draws(Xk, yk, X, L, ai_idx, al, j, B_PB)
      qd <- quantile(d, c((1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2), names = FALSE)
      base <- c(rep = rep_id, model = mi, n_known = nk, n_ai = n_ai, alpha = al, truth = truth)
      out[[length(out) + 1]] <- c(base, method = 1, mean = ci[[1]], lo = ci[[2]], hi = ci[[3]],
                                  cal_cov = cov_pp[a], cv_lpd = cv, test_lpd = te_lpd, test_rmse = te_rmse,
                                  lambda = NA)
      out[[length(out) + 1]] <- c(base, method = 2, mean = mean(d), lo = qd[1], hi = qd[2],
                                  cal_cov = cov_pb[a], cv_lpd = NA, test_lpd = NA, test_rmse = NA,
                                  lambda = NA)
    }

    # PPI / PPI++ with the same labelled and unlabelled students (needs LLM predictions for both)
    fk <- PRED[[md$y]][known_idx]; fu <- PRED[[md$y]][ai_idx]
    if (HAS_PPI && !anyNA(fk) && !anyNA(fu)) {
      pr <- ppi(Xk, yk, fk, Xa, fu, j)
      for (k in 1:2) {
        r <- pr[[k]]
        out[[length(out) + 1]] <- c(rep = rep_id, model = mi, n_known = nk, n_ai = n_ai, alpha = 0,
                                    truth = truth, method = 2 + k, mean = r[["mean"]], lo = r[["lo"]],
                                    hi = r[["hi"]], cal_cov = NA, cv_lpd = NA, test_lpd = NA,
                                    test_rmse = NA, lambda = r[["lambda"]])
      }
    }
  }
  do.call(rbind, out)
}

finish <- function(mat) {
  d <- as.data.frame(mat)
  d$model  <- names(models)[d$model]
  d$method <- c("power", "pb", "ppi", "ppi_pp")[d$method]
  key <- paste(d$rep, d$model, d$n_known, d$method)
  d$cv_best <- d$method == "power" &
    as.logical(ave(ifelse(is.na(d$cv_lpd), -Inf, d$cv_lpd), key, FUN = function(v) v == max(v)))
  d$cal_best <- d$method %in% c("power", "pb") & as.logical(ave(seq_len(nrow(d)), key, FUN = function(i) {
    ok <- !is.na(d$cal_cov[i]) & d$cal_cov[i] >= LEVEL
    a  <- if (any(ok)) max(d$alpha[i][ok]) else 0
    d$alpha[i] == a
  }))
  d$pooled <- d$alpha == d$n_ai
  d$covers <- d$lo <= d$truth & d$truth <= d$hi
  d
}
SETTINGS <- list(                                    # named selection rules
  known_only = function(d) d$method == "power" & d$alpha == 0,
  pooled_w1  = function(d) d$method == "power" & d$pooled,
  power_cv   = function(d) d$method == "power" & d$cv_best,
  power_cal  = function(d) d$method == "power" & d$cal_best,
  pb_alpha0  = function(d) d$method == "pb" & d$alpha == 0,
  pb_pooled  = function(d) d$method == "pb" & d$pooled,
  pb_cal     = function(d) d$method == "pb" & d$cal_best,
  ppi        = function(d) d$method == "ppi",
  ppi_pp     = function(d) d$method == "ppi_pp")

# --------------------------------------------------------------------------
# 5. A. Saved split
# --------------------------------------------------------------------------
set.seed(SEED)
known_saved <- match(split$known_ids, data$student_id)
cat("A. saved split ...\n")
A <- finish(do.call(rbind, lapply(N_KNOWN_SET, function(nk) {
  kn <- known_saved[seq_len(nk)]
  analyse(kn, llm_rows, setdiff(seq_len(N), c(kn, llm_rows)), 0)
})))
write.csv(A, file.path(OUT_DIR, "single_split_grid.csv"), row.names = FALSE)

fmt <- function(r) if (nrow(r)) sprintf("%.3g [%.3g, %.3g]%s", r$mean[1], r$lo[1], r$hi[1],
                                         if (r$alpha[1] %in% c(0, r$n_ai[1])) "" else sprintf(" a=%g", r$alpha[1])) else NA
summA <- do.call(rbind, lapply(split(A, list(A$model, A$n_known), drop = TRUE), function(s) {
  tf <- lm.fit(models[[s$model[1]]]$X[setdiff(seq_len(N), known_saved[seq_len(s$n_known[1])]), ],
               data[[models[[s$model[1]]]$y]][setdiff(seq_len(N), known_saved[seq_len(s$n_known[1])])])
  cbind(data.frame(model = s$model[1], n_known = s$n_known[1], truth = signif(s$truth[1], 3)),
        as.data.frame(lapply(SETTINGS, function(f) fmt(s[f(s), ]))))
}))
summA <- summA[order(match(summA$model, names(models)), -summA$n_known), ]
write.csv(summA, file.path(OUT_DIR, "single_split_summary.csv"), row.names = FALSE)
old <- options(width = 250)
cat("\n===== A. Saved split: beta(ai_hours) mean [", LEVEL * 100, "% interval] (a = chosen alpha) =====\n")
print(summA, row.names = FALSE, right = FALSE)
options(old)

# check: Theorem-1 approximation vs real posterior-bootstrap draws (used inside the calibration)
chk <- do.call(rbind, lapply(names(models), function(mn) {
  md <- models[[mn]]; kn <- known_saved
  aiX <- list(X = md$X[llm_rows, ], m1 = LLM[[md$y]]$m1[llm_rows], m2 = LLM[[md$y]]$m2[llm_rows])
  aiX$XX <- crossprod(aiX$X) / length(llm_rows); aiX$Xm1 <- crossprod(aiX$X, aiX$m1) / length(llm_rows)
  do.call(rbind, lapply(c(0, 30, 300), function(al) {
    s <- A[A$model == mn & A$n_known == 300 & A$method == "pb" & A$alpha == al, ]
    ap <- pb_asym(md$X[kn, ], data[[md$y]][kn], aiX, al, md$j)
    data.frame(model = mn, alpha = al, pb_lo = s$lo, pb_hi = s$hi, theorem1_lo = ap[[2]], theorem1_hi = ap[[3]])
  }))
}))
cat("\nTheorem-1 approximation vs posterior-bootstrap draws (n_known = 300):\n")
print(chk, digits = 3, row.names = FALSE)
write.csv(chk, file.path(OUT_DIR, "theorem1_check.csv"), row.names = FALSE)

# --------------------------------------------------------------------------
# 6. B. Replicated random splits
# --------------------------------------------------------------------------
cat("\nB. running", N_REP, "random splits ...\n")
t0 <- Sys.time()
B <- vector("list", N_REP)
for (r in seq_len(N_REP)) {
  set.seed(SEED + r)
  ai_idx <- sample(llm_rows, N_AI_REP)
  pool   <- sample(setdiff(seq_len(N), ai_idx))
  B[[r]] <- do.call(rbind, lapply(N_KNOWN_SET, function(nk) {
    kn <- pool[seq_len(nk)]                                # nested: 100 known rows are inside the 300
    analyse(kn, ai_idx, setdiff(pool, kn), r)
  }))
  if (r %% 10 == 0) cat(sprintf("  %d / %d  (%.1f min)\n", r, N_REP, as.numeric(Sys.time() - t0, units = "mins")))
}
B <- finish(do.call(rbind, B))
write.csv(B, file.path(OUT_DIR, "replicates_raw.csv"), row.names = FALSE)

metrics <- function(s) data.frame(
  n = nrow(s), mean_alpha = mean(s$alpha), mean_lambda = mean(s$lambda),
  bias = mean(s$mean - s$truth), rmse = sqrt(mean((s$mean - s$truth)^2)),
  coverage = mean(s$covers), width = mean(s$hi - s$lo),
  sign_ok = mean(sign(s$mean) == sign(s$truth)),
  test_lpd = mean(s$test_lpd), test_rmse = mean(s$test_rmse))
grp <- split(B, list(B$model, B$n_known), drop = TRUE)
rep_sel <- do.call(rbind, lapply(grp, function(s)
  cbind(model = s$model[1], n_known = s$n_known[1],
        do.call(rbind, lapply(names(SETTINGS), function(nm) cbind(setting = nm, metrics(s[SETTINGS[[nm]](s), ])))))))
rep_sel <- rep_sel[order(match(rep_sel$model, names(models)), -rep_sel$n_known), ]
write.csv(rep_sel, file.path(OUT_DIR, "replicates_summary.csv"), row.names = FALSE)

rep_grid <- aggregate(cbind(coverage = covers, cal_cov, rmse2 = (mean - truth)^2, width = hi - lo) ~
                        model + n_known + method + alpha, data = B, FUN = mean)
rep_grid$rmse <- sqrt(rep_grid$rmse2); rep_grid$rmse2 <- NULL
write.csv(rep_grid, file.path(OUT_DIR, "replicates_by_alpha.csv"), row.names = FALSE)

cal_dist <- with(B[B$cal_best & B$method %in% c("power", "pb"), ], table(paste(model, n_known, method), alpha))
write.csv(as.data.frame.matrix(cal_dist), file.path(OUT_DIR, "replicates_cal_alpha_chosen.csv"))

show <- rep_sel; num <- sapply(show, is.numeric); show[num] <- lapply(show[num], round, 4)
old <- options(width = 250)
cat("\n=====", N_REP, "random splits (", N_AI_REP, "AI rows each): beta(ai_hours) vs truth on unseen rows,",
    LEVEL * 100, "% intervals =====\n")
print(show[, c("model", "n_known", "setting", "mean_alpha", "mean_lambda", "bias", "rmse", "coverage", "width", "sign_ok", "test_lpd")],
      row.names = FALSE)
cat("\nalpha chosen by bootstrap coverage calibration (counts over splits):\n"); print(cal_dist)
options(old)

# --------------------------------------------------------------------------
# 7. Plots: drawn from the saved tables by make_figures.R (shared legend below the panels), so they
#    can also be redrawn later without re-running the analysis
# --------------------------------------------------------------------------
source(file.path("regression", "make_figures.R"))
make_figures(OUT_DIR, level = LEVEL)

cat("\nOutputs in", normalizePath(OUT_DIR), ":", paste(list.files(OUT_DIR), collapse = ", "), "\n")
