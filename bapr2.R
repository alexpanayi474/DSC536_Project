#WANT TO CHECK HOW AI HOURS PER WEEK REFLECT
#1. GPA SCORE
#2. CRITICAL THINKING SCORE
#LOOK HOW USING AI LABELLED PRIOR DATA INFLUENCES THE FINAL RESULT

#==========================================
# 0. SETUP & ENVIRONMENT CLEANING
# ==========================================
rm(list=ls())
gc()

library(dplyr)
library(httr)
library(jsonlite)

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
save_path <- "C:/Users/stefa/Desktop/FILELOCALI/CIPRO-LEZIONIFILE/babaababab/" #put in yours
data <- read.csv(paste0(save_path, "campus_ai_pulse_2026.csv"), stringsAsFactors = TRUE)
data <- data[, -c(1, 9, 13, 14, 19, 20, 21, 22, 23, 24)]
#data <- na.omit(data), not needed as there are no NAs

# ==========================================
# 2. DATASET SLICING
# ==========================================

n_total_rows <- nrow(data)
n_known <- 300
n_ai    <- 900

data_shuffled <- data[sample(n_total_rows), ] #random sampling from data to divide labelled and unllabelled
data_known <- data_shuffled %>% slice(1:n_known)

#rearranging for ai
data_ai_raw <- data_shuffled %>%
  slice((n_known + 1):(n_known + n_ai)) %>%
  dplyr::select(-cgpa, -critical_thinking_score)

set.seed(30175)

# ==========================================
# 4. COMPACT JSON BATCH PROMPT FUNCTION (LOW-TOKEN)
# ==========================================
create_batch_prompt <- function(data_batch) {
  feature_cols <- setdiff(names(data_batch), "batch_idx")
  header <- paste(feature_cols, collapse = "|")
  
  students_text <- apply(data_batch, 1, function(row) {
    paste0("S", trimws(row["batch_idx"]), ":", paste(trimws(row[feature_cols]), collapse = "|"))
  })
  
  #prompt used for all obs.
  prompt_text <- paste0(
    "You are an educational data analyst. Estimate two probability distributions (each summing to 1.0) per student:\n",
    "1) 'cgpa': 8 bins [0.0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2.0-2.5, 2.5-3.0, 3.0-3.5, 3.5-4.0]\n",
    "2) 'score': 10 bins for Critical Thinking Score [0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100]\n",
    "Columns: ", header, "\n",
    "Return ONLY a valid JSON object with key 'students' (one item per student in order), no extra text:\n",
    "{\"students\":[{\"id\":\"S1\",\"cgpa\":[0,0,0.02,0.08,0.2,0.35,0.25,0.1],\"score\":[0,0,0.02,0.05,0.13,0.25,0.3,0.15,0.08,0.02]}]}\n\n",
    paste(students_text, collapse = "\n")
  )
  return(prompt_text)
}

# ==========================================
# 5. GROQ API CONFIGURATION & FUNCTION
# ==========================================

#AI KEY, MODEL AND  ENDPOINT
API_KEY  <- "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"
MODEL_ID <- "openai/gpt-oss-20b"
ENDPOINT <- "https://api.groq.com/openai/v1/chat/completions"

get_response <- function(prompt_text) {
  payload <- list(
    model = MODEL_ID,
    messages = list(list(role = "user", content = prompt_text)),
    temperature = 0.4,
    max_tokens = 2500,
    reasoning_effort = "low"
  )
  body_json <- toJSON(payload, auto_unbox = TRUE)
  
  for (attempt in 1:5) {
    response <- tryCatch({
      POST(
        url = ENDPOINT,
        add_headers(
          "Authorization" = paste("Bearer", API_KEY),
          "Content-Type" = "application/json"
        ),
        body = body_json,
        encode = "raw"
      )
    }, error = function(e) {
      cat("  [Errore connessione]:", e$message, "\n")
      NULL
    })
    
    if (!is.null(response)) {
      sc <- status_code(response)
      if (sc == 200) {
        result <- content(response, "parsed", simplifyVector = FALSE)
        txt <- tryCatch(result$choices[[1]]$message$content, error = function(e) "")
        if (!is.null(txt) && nchar(trimws(txt)) > 0) return(txt)
      } else if (sc == 429) {
        err_msg <- content(response, "text", encoding = "UTF-8")
        if (grepl("tokens per day|TPD", err_msg, ignore.case = TRUE)) {
          return("TPD_LIMIT")
        }
        cat("  [Rate Limit TPM/RPM] Pausing 12s...\n")
        Sys.sleep(12)
      } else {
        err_msg <- content(response, "text", encoding = "UTF-8")
        cat("  [HTTP", sc, "]:", err_msg, "\n")
        Sys.sleep(2)
      }
    }
  }
  return(NA)
}

# ==========================================
# 6. EXECUTE BATCHES WITH AUTO-RESUME FROM BACKUP
# ==========================================
batch_size  <- 15
n_total     <- nrow(data_ai_raw)
n_batches   <- ceiling(n_total / batch_size)
file_final  <- paste0(save_path, "risposte_groq_oss20b.rds")
file_backup <- paste0(save_path, "backup_groq_oss20b.rds")

#if files are imported correctly it should just say Ripristinato file finale esistente
#which basically means we already got the file

if (file.exists(file_final)) {
  batch_responses <- readRDS(file_final)
  cat("Ripristinato file finale esistente:", sum(!is.na(batch_responses)), "/", n_batches, "batch completati.\n")
} else if (file.exists(file_backup)) {
  batch_responses <- readRDS(file_backup)
  cat("Ripristinato backup esistente:", sum(!is.na(batch_responses)), "/", n_batches, "batch completati.\n")
} else if (exists("batch_responses") && length(batch_responses) == n_batches) {
  cat("Uso batch_responses già in memoria:", sum(!is.na(batch_responses)), "/", n_batches, "batch completati.\n")
} else {
  batch_responses <- rep(NA_character_, n_batches)
  cat("Nessun backup trovato. Avvio da zero.\n")
}


length(batch_responses) <- n_batches

for (b in 1:n_batches) {
  if (!is.na(batch_responses[b]) && nchar(trimws(batch_responses[b])) > 20) next
  
  start_idx <- (b - 1) * batch_size + 1
  end_idx   <- min(b * batch_size, n_total)
  
  data_sub <- data_ai_raw[start_idx:end_idx, ]
  data_sub$batch_idx <- 1:nrow(data_sub)
  
  cat("Batch", b, "/", n_batches, "(rows", start_idx, "to", end_idx, ")...\n")
  res_b <- get_response(create_batch_prompt(data_sub))
  
  if (!is.na(res_b) && res_b == "TPD_LIMIT") {
    saveRDS(batch_responses, file_backup)
    stop("Daily token limit (TPD) reached. Progressi salvati in ", file_backup, ". Rilancia domani per riprendere da batch ", b, ".")
  }
  
  batch_responses[b] <- res_b
  
  if (b %% 5 == 0) {
    saveRDS(batch_responses, file_backup)
  }
  Sys.sleep(2.5)
}

saveRDS(batch_responses, file_final)
cat("Tutti i batch completati e salvati in:", file_final, "\n")

# ==========================================
# 7. JSON PARSE, REPAIR & SAMPLE FUNCTION
# ==========================================
build_ai_dataset <- function(data_ai_raw, save_path, batch_size = 15, max_repairs = 3, seed = 30175) {
  n_ai <- nrow(data_ai_raw)
  file_rds <- paste0(save_path, "risposte_groq_oss20b.rds")
  if (!file.exists(file_rds)) stop("Raw responses file not found: ", file_rds)
  
  raw_responses <- readRDS(file_rds)
  
  gpa_lows    <- seq(0.0, 3.5, by = 0.5)
  gpa_highs   <- seq(0.5, 4.0, by = 0.5)
  score_lows  <- seq(0, 90, by = 10)
  score_highs <- seq(10, 100, by = 10)
  
  parse_vectors <- function(responses) {
    set.seed(seed)
    sampled_gpa   <- rep(NA_real_, n_ai)
    sampled_score <- rep(NA_real_, n_ai)
    
    for (b in seq_along(responses)) {
      if (is.na(responses[b]) || nchar(trimws(responses[b])) == 0) next
      
      clean_txt <- regmatches(responses[b], regexpr("\\{[\\s\\S]*\\}", responses[b], perl = TRUE))
      if (length(clean_txt) == 0) next
      
      parsed_json <- tryCatch(fromJSON(clean_txt, simplifyVector = FALSE), error = function(e) NULL)
      if (is.null(parsed_json) || is.null(parsed_json$students)) next
      
      students_list <- parsed_json$students
      for (j in seq_len(min(length(students_list), batch_size))) {
        global_idx <- (b - 1) * batch_size + j
        if (global_idx > n_ai) break
        
        st <- students_list[[j]]
        probs_gpa   <- as.numeric(unlist(st$cgpa))
        probs_score <- as.numeric(unlist(st$score))
        
        if (length(probs_gpa) == 8 && length(probs_score) == 10 &&
            !any(is.na(probs_gpa)) && !any(is.na(probs_score)) &&
            sum(probs_gpa) > 0 && sum(probs_score) > 0) {
          
          p_g <- pmax(0, probs_gpa) / sum(pmax(0, probs_gpa))
          bin_g <- sample(1:8, size = 1, prob = p_g)
          sampled_gpa[global_idx] <- round(runif(1, gpa_lows[bin_g], gpa_highs[bin_g]), 2)
          
          p_s <- pmax(0, probs_score) / sum(pmax(0, probs_score))
          bin_s <- sample(1:10, size = 1, prob = p_s)
          sampled_score[global_idx] <- round(runif(1, score_lows[bin_s], score_highs[bin_s]), 1)
        }
      }
    }
    return(list(gpa = sampled_gpa, score = sampled_score))
  }
  
  parsed <- parse_vectors(raw_responses)
  
  for (pass in seq_len(max_repairs)) {
    bad_idx <- which(is.na(parsed$gpa) | is.na(parsed$score))
    bad_batches <- unique(ceiling(bad_idx / batch_size))
    if (length(bad_batches) == 0) break
    
    cat("Pass", pass, "- Repairing", length(bad_batches), "batch(es):", paste(bad_batches, collapse = ", "), "\n")
    
    for (b in bad_batches) {
      start_idx <- (b - 1) * batch_size + 1
      end_idx   <- min(b * batch_size, n_ai)
      
      data_sub <- data_ai_raw[start_idx:end_idx, ]
      data_sub$batch_idx <- 1:nrow(data_sub)
      
      res_b <- get_response(create_batch_prompt(data_sub))
      if (!is.na(res_b) && res_b == "TPD_LIMIT") {
        saveRDS(raw_responses, file_rds)
        stop("Daily token limit (TPD) reached during repair. Progressi salvati.")
      }
      raw_responses[b] <- res_b
      Sys.sleep(2.5)
    }
    
    saveRDS(raw_responses, file_rds)
    parsed <- parse_vectors(raw_responses)
  }
  
  data_ai_complete <- data_ai_raw
  data_ai_complete$cgpa <- parsed$gpa
  data_ai_complete$critical_thinking_score <- parsed$score
  
  saveRDS(data_ai_complete, paste0(save_path, "data_ai_campus_complete.rds"))
  return(data_ai_complete)
}

#now we should have the right datasets
data_ai_complete <- build_ai_dataset(data_ai_raw, save_path, batch_size = 15)

# ==========================================
# 8. MERGE DATASETS & RE-ESTIMATE MODELS
# ==========================================
#estimating parameters
data_combined <- bind_rows(data_known, data_ai_complete)

m_truth_score_all  <- lm(critical_thinking_score ~ . - cgpa, data = data) 
c_truth_score_all  <- coef(m_truth_score_all)["ai_hours_per_week"]

m_truth_score_only <- lm(critical_thinking_score ~ ai_hours_per_week, data = data)
c_truth_score_only <- coef(m_truth_score_only)["ai_hours_per_week"]

m_truth_gpa_all    <- lm(cgpa ~ . - critical_thinking_score, data = data)
c_truth_gpa_all    <- coef(m_truth_gpa_all)["ai_hours_per_week"]

m_truth_gpa_only   <- lm(cgpa ~ ai_hours_per_week, data = data)
c_truth_gpa_only   <- coef(m_truth_gpa_only)["ai_hours_per_week"]

# --- MODELLI SU DATASET RIUNITO (Known + AI) ---
m_comb_score_all  <- lm(critical_thinking_score ~ . - cgpa, data = data_combined)
c_comb_score_all  <- coef(m_comb_score_all)["ai_hours_per_week"]

m_comb_score_only <- lm(critical_thinking_score ~ ai_hours_per_week, data = data_combined)
c_comb_score_only <- coef(m_comb_score_only)["ai_hours_per_week"]

m_comb_gpa_all    <- lm(cgpa ~ . - critical_thinking_score, data = data_combined)
c_comb_gpa_all    <- coef(m_comb_gpa_all)["ai_hours_per_week"]

m_comb_gpa_only   <- lm(cgpa ~ ai_hours_per_week, data = data_combined)
c_comb_gpa_only   <- coef(m_comb_gpa_only)["ai_hours_per_week"]

# --- MODELLI SOLO SU KNOWN ---
m_known_score_all  <- lm(critical_thinking_score ~ . - cgpa, data = data_known)
c_known_score_all  <- coef(m_known_score_all)["ai_hours_per_week"]

m_known_score_only <- lm(critical_thinking_score ~ ai_hours_per_week, data = data_known)
c_known_score_only <- coef(m_known_score_only)["ai_hours_per_week"]

m_known_gpa_all    <- lm(cgpa ~ . - critical_thinking_score, data = data_known)
c_known_gpa_all    <- coef(m_known_gpa_all)["ai_hours_per_week"]

m_known_gpa_only   <- lm(cgpa ~ ai_hours_per_week, data = data_known)
c_known_gpa_only   <- coef(m_known_gpa_only)["ai_hours_per_week"]

# --- MODELLI SOLO SU AI ---
m_ai_score_all  <- lm(critical_thinking_score ~ . - cgpa, data = data_ai_complete)
c_ai_score_all  <- coef(m_ai_score_all)["ai_hours_per_week"]

m_ai_score_only <- lm(critical_thinking_score ~ ai_hours_per_week, data = data_ai_complete)
c_ai_score_only <- coef(m_ai_score_only)["ai_hours_per_week"]

m_ai_gpa_all    <- lm(cgpa ~ . - critical_thinking_score, data = data_ai_complete)
c_ai_gpa_all    <- coef(m_ai_gpa_all)["ai_hours_per_week"]

m_ai_gpa_only   <- lm(cgpa ~ ai_hours_per_week, data = data_ai_complete)
c_ai_gpa_only   <- coef(m_ai_gpa_only)["ai_hours_per_week"]

results_comparison <- data.frame(
  Modello = c("Score ~ All", "Score ~ Only AI Hours", "CGPA ~ All", "CGPA ~ Only AI Hours"),
  Truth_Full = c(c_truth_score_all, c_truth_score_only, c_truth_gpa_all, c_truth_gpa_only),
  Only_Known = c(c_known_score_all, c_known_score_only, c_known_gpa_all, c_known_gpa_only),
  Only_AI = c(c_ai_score_all, c_ai_score_only, c_ai_gpa_all, c_ai_gpa_only),
  Combined_Known_Plus_AI = c(c_comb_score_all, c_comb_score_only, c_comb_gpa_all, c_comb_gpa_only)
)

print(results_comparison)

#basically
#CRITICAL THINKING SCORE: the known data overestimate the negative impact
#ai however underestimate it (quite ironic) but its closer to the truth
#all in all, the combined solution is the best one
#GPA: in the full model, both ai and known data are lower
#combining them however gives out an even lower outcome 
#very weird, should go into it a bit furtheer
#singlevariate model gives out a more expected solution
#known data slightly overestimate, ai the opposite (even different signs)
#however, the combined solution is very close to the truth