# Build report/report.pdf from report/report.md (run from the project root in RStudio):
#   source("report/build_pdf.R")
# Needs TinyTeX (already installed: tinytex::is_tinytex() is TRUE) and pandoc (comes with RStudio).
# Page size, margins and font are set in the YAML header of report.md.

report_dir <- normalizePath("report")      # figure paths in report.md are relative to report/
rmarkdown::pandoc_convert("report.md", to = "pdf", output = "report.pdf", wd = report_dir,
                          options = c("--pdf-engine=xelatex", "--standalone"))
cat("Written:", file.path(report_dir, "report.pdf"), "\n")
