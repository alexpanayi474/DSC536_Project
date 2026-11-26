# ==========================================================================
# make_figures.R -- draw the result figures from the saved tables of regression_power_prior.R
# (single_split_grid.csv, replicates_by_alpha.csv, replicates_summary.csv), so the figures can be
# redrawn without re-running the analysis. One shared legend is placed BELOW the four panels, so
# it never covers the data. Called automatically at the end of regression_power_prior.R.
#
# Redraw by hand (from the project root):
#   source("regression/make_figures.R"); make_figures("regression/output_v3")
# ==========================================================================

make_figures <- function(out_dir, level = 0.90) {
  A  <- read.csv(file.path(out_dir, "single_split_grid.csv"))
  rg <- read.csv(file.path(out_dir, "replicates_by_alpha.csv"))
  rs <- read.csv(file.path(out_dir, "replicates_summary.csv"))
  A  <- A[A$method %in% c("power", "pb"), ]          # PPI rows have no alpha
  rg <- rg[rg$method %in% c("power", "pb"), ]
  n_rep  <- if ("n" %in% names(rs)) max(rs$n, na.rm = TRUE) else NA
  models <- intersect(c("score_only", "score_all", "cgpa_only", "cgpa_all"), unique(A$model))
  title  <- c(score_only = "Critical thinking ~ AI hours", score_all = "Critical thinking ~ all covariates",
              cgpa_only = "CGPA ~ AI hours", cgpa_all = "CGPA ~ all covariates")
  cols   <- c(power = "#1f77b4", pb = "#d95f02")
  alab   <- function(a) ifelse(a >= 1000, paste0(a / 1000, "k"), as.character(a))
  nks    <- sort(unique(A$n_known), decreasing = TRUE)

  open_png <- function(name) {
    png(file.path(out_dir, name), width = 1400, height = 1180, res = 120)
    par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3, 1), oma = c(3.2, 0, 0, 0))
  }
  legend_below <- function(...) {                     # one legend for the whole figure, outside the panels
    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
    plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n", xlab = "", ylab = "")
    legend("bottom", horiz = TRUE, bty = "n", xpd = TRUE, cex = 0.95, ...)
  }

  # 1. Posterior of beta(ai_hours) against alpha, saved split
  for (nk in nks) {
    open_png(sprintf("beta_ai_vs_alpha_n%d.png", nk))
    for (mn in models) {
      s <- A[A$model == mn & A$n_known == nk, ]; al <- sort(unique(s$alpha)); xp <- seq_along(al)
      plot(NA, xlim = c(0.6, length(al) + 0.4), ylim = range(c(s$lo, s$hi, s$truth)), xaxt = "n",
           xlab = "alpha = effective number of AI data points", ylab = "coefficient of AI hours",
           main = sprintf("%s, %d labelled", title[mn], nk))
      axis(1, xp, alab(al)); abline(h = s$truth[1], lty = 2); abline(h = 0, col = "grey60")
      for (me in c("power", "pb")) {
        q <- s[s$method == me, ]; q <- q[order(q$alpha), ]; off <- if (me == "power") -0.12 else 0.12
        arrows(xp + off, q$lo, xp + off, q$hi, angle = 90, code = 3, length = 0.03, col = cols[me])
        points(xp + off, q$mean, pch = 16, col = cols[me])
        sel <- as.logical(q$cal_best); points((xp + off)[sel], q$mean[sel], pch = 2, cex = 2, col = cols[me])
        if (me == "power") { cv <- as.logical(q$cv_best); points((xp + off)[cv], q$mean[cv], pch = 8, cex = 2, col = cols[me]) }
      }
    }
    legend_below(legend = c("power prior", "posterior bootstrap (DP prior)", "truth", "alpha chosen by CV",
                            "alpha chosen by calibration"),
                 col = c(cols, "black", "black", "black"), pch = c(16, 16, NA, 8, 2), lty = c(NA, NA, 2, NA, NA))
    dev.off()
  }

  # 2. Actual coverage over the random splits vs the bootstrap estimate used by calibration (paper Fig. 3)
  for (nk in nks) {
    open_png(sprintf("calibration_n%d.png", nk))
    for (mn in models) {
      s <- rg[rg$model == mn & rg$n_known == nk, ]; al <- sort(unique(s$alpha)); xp <- seq_along(al)
      plot(NA, xlim = range(xp), ylim = c(0, 1), xaxt = "n", xlab = "alpha", ylab = "coverage of 90% interval",
           main = sprintf("%s, %d labelled%s", title[mn], nk, if (is.na(n_rep)) "" else sprintf(", %d splits", n_rep)))
      axis(1, xp, alab(al)); abline(h = level, col = "grey40", lty = 3)
      for (me in c("power", "pb")) {
        q <- s[s$method == me, ]; q <- q[order(q$alpha), ]
        lines(xp, q$coverage, type = "b", pch = 16, col = cols[me])
        lines(xp, q$cal_cov, type = "b", pch = 1, lty = 2, col = cols[me])
      }
    }
    legend_below(legend = c("power prior: actual", "power prior: bootstrap estimate",
                            "posterior bootstrap: actual", "posterior bootstrap: bootstrap estimate"),
                 col = rep(cols, each = 2), lty = c(1, 2, 1, 2), pch = c(16, 1, 16, 1))
    dev.off()
  }

  # 3. RMSE over the random splits against alpha
  open_png("replicates_rmse_by_alpha.png")
  for (mn in models) {
    s <- rg[rg$model == mn, ]; al <- sort(unique(s$alpha)); xp <- seq_along(al)
    plot(NA, xlim = range(xp), ylim = range(s$rmse), xaxt = "n", xlab = "alpha",
         ylab = "RMSE of the coefficient of AI hours",
         main = sprintf("%s%s", title[mn], if (is.na(n_rep)) "" else sprintf(", %d splits", n_rep)))
    axis(1, xp, alab(al))
    for (me in c("power", "pb")) for (nk in nks) {
      q <- s[s$method == me & s$n_known == nk, ]; q <- q[order(q$alpha), ]
      lines(xp, q$rmse, type = "b", pch = 16, lty = if (nk == max(nks)) 1 else 2, col = cols[me])
    }
  }
  legend_below(legend = c(sprintf("power prior, %d labelled", nks), sprintf("posterior bootstrap, %d labelled", nks)),
               col = rep(cols, each = length(nks)), lty = rep(c(1, 2), 2), pch = 16)
  dev.off()
  invisible(list.files(out_dir, pattern = "\\.png$"))
}
