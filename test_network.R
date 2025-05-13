rm(list=ls())
source("nn_functions.R")

#### Naming ----

# t = target (i.e., picture)
# d = distractor (i.e., word)

# H = hidden
# O = output

# w = network weight
# b = bias weight
# a = unit activation
# n = net input

#### Constants ----

# Default weights as matrices
wHt <- make_weight_matrix(pos = 3, neg = -3)
wHd <- make_weight_matrix(pos = 3.5, neg = -3.5)
wOt <- make_weight_matrix(pos = 1.5, neg = -1.5)
wOd <- make_weight_matrix(pos = 1.5, neg = -1.5)

# Default bias vectors
bHt <- c(-4, -4, -4, -4)
bHd <- c(-4, -4, -4, -4) 

# Task weights (only distractor gets used.)
wTt <- c(2, 2, 2, 2)
wTd <- c(2, 2, 2, 2)

# Item lookup table for all stimuli (lower case is target, 
# upper case is distractor)
stim_lookup <- list(
  dog  = c(1,0,0,0),
  DOG  = c(1,0,0,0),
  bird = c(0,1,0,0),
  BIRD = c(0,1,0,0),
  cat  = c(0,0,1,0),
  CAT  = c(0,0,1,0),
  fish = c(0,0,0,1),
  FISH = c(0,0,0,1)
)


#### Construct initial network ----

# Initialization of transient values

# Activation
aO <- nOd <- nOt <- aHd <- aHt <- nHt <- nHd <- bHt <- rep(0, 4)

# Control
control <- c(dog = 0, bird = 0, cat = 0, fish = 0)

# Make network list
nn <- list(wOt=wOt,wOd=wOd, # Output weight matrices
           wHt=wHt,wHd=wHd, # Hidden weight matrices
           bHt=bHt,bHd=bHd, # Hidden bias vectors
           wTd=wTd,wTt=wTt, # Task weight vector
           activation_fun=plogis)  # Hidden output function

state <- list(aO=aO,           # Output activation vector
              nOt=nOt,nOd=nOd, # Output net vectors
              aHd=aHd,aHt=aHt, # Hidden activation vectors
              nHt=nHt,nHd=nHd, # Hidden net vectors
              control=control) # Control values
state0 <- state


#### Run a single trial ----

# Congruent trial
stimulus <- list(target = "dog", distractor = "DOG")
between_update(stimulus, state, nn, stim_lookup = stim_lookup, print_conflict = TRUE)

# Incongruent trial
stimulus <- list(target = "cat", distractor = "FISH")
between_update(stimulus, state, nn, stim_lookup = stim_lookup, print_conflict = TRUE)


#### Simulate a trial sequence with proportion-congruency effect ----

# Make distractors
set.seed(8)
n_per_target <- 500

# Define targets
targets <- rep(c("dog", "bird", "cat", "fish"), each = n_per_target)

# Define item-specific congruency rates
cong_probs <- c(dog = 0.80, bird = 0.20, cat = 0.50, fish = 0.90)

# Generate distractors based on item-specific congruency
distractors <- character(length(targets))
for (i in seq_along(targets)) {
  tgt <- targets[i]
  prob_cong <- cong_probs[tgt]
  
  if (runif(1) < prob_cong) {
    # Congruent: distractor matches target
    distractors[i] <- toupper(tgt)
  } else {
    # Incongruent: randomly draw a non-target item
    distractors[i] <- sample(setdiff(toupper(names(cong_probs)), toupper(tgt)), 1)
  }
}

stim_sequence <- data.frame(
  target = targets,
  distractor = distractors,
  stringsAsFactors = FALSE
)
str(stim_sequence)
stim_sequence$congruent <- tolower(stim_sequence$target) == tolower(stim_sequence$distractor)
round(prop.table(table(stim_sequence$target, stim_sequence$congruent), margin = 1), 2)

# Run network on trial sequence
states <- run_between_sequence(stim_sequence, state0, nn, stim_lookup)
states

# Skip the initial state (index 1), extract aO from each trial
activation_list <- lapply(states[-1], function(s) s$aO)
control_list <- lapply(states[-1], function(s) s$control)
save(activation_list, control_list, file = "data/activation_list.RData")


# LBA parameters (true values for simulation)
v_base <- 1
A <- 0.5
B <- 0.4
b <- A + B
t0 <- 0.3
sv <- 0.2

sim_full <- simulate_lba_from_states(
  states,
  stim_sequence,
  v_base = v_base,
  A = A,
  B = B,
  t0 = t0,
  sv = sv
)
head(sim_full)
tail(sim_full)
summary(sim_full$correct)
table(sim_full$congruent)
mean(sim_full$correct)
table(sim_full$target, sim_full$congruent)
round(prop.table(table(sim_full$target, sim_full$congruent), margin = 1), 2)

dat <- sim_full
# Save
save(dat, file = "data/sim_dat.RData")


# Plot --------------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(tidyr)
library(zoo)
library(patchwork)

unit_colors <- c(
  "dog"  = "#1b9e77",
  "bird" = "#d95f02",
  "cat"  = "#7570b3",
  "fish" = "#e7298a"
)

unit_levels <- c("dog", "bird", "cat", "fish")

# Compute accuracy
df <- dat %>%
  mutate(
    acc_numeric = as.numeric(correct),
    acc_ma = rollmean(acc_numeric, k = 5, fill = NA, align = "center")
  )

response_labels <- c("dog", "bird", "cat", "fish")
df$response_label <- factor(response_labels[df$response],
                                       levels = names(unit_colors))

# Rectangles for background bands
band_df <- df %>%
  transmute(
    xmin = trial - 0.5,
    xmax = trial + 0.5,
    ymin = -Inf,
    ymax = Inf,
    congruent = congruent
  )

mean_acc <- mean(df$acc_numeric, na.rm = TRUE)

# Plot accuracy
p_acc <- ggplot(df, aes(x = trial)) +
  geom_rect(data = band_df,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = congruent),
            alpha = 0.15, inherit.aes = FALSE) +
  geom_point(aes(y = acc_numeric, color = response_label), size = 2.5, alpha = 0.8) +  # 🔁 changed from `correct` to `target`
  geom_line(aes(y = acc_numeric), alpha = 0.3) +
  geom_line(aes(y = acc_ma), color = "black", linewidth = 1, linetype = "solid", alpha = 0.5) +
  geom_hline(yintercept = mean_acc, color = "black", linetype = "dashed", linewidth = 1, alpha = 0.5) +
  annotate("text",
           x = -Inf, y = -Inf,
           label = paste0("Mean acc = ", round(mean_acc, 2)),
           hjust = -0.4, vjust = -2.0,
           size = 4.5) +
  scale_color_manual(values = unit_colors) +  # 🔁 match RT/activation plots
  scale_fill_manual(values = c(`TRUE` = "white", `FALSE` = "lightcoral"), guide = "none") +
  scale_y_continuous(limits = c(0, 1), breaks = c(0, 1)) +
  labs(title = "Accuracy with Moving Average", x = "Trial",
       y = "Correct", color = "Response") +
  theme_minimal()

# Plot RT
p_rt <- ggplot(df, aes(x = trial)) +
  geom_rect(data = band_df,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = congruent),
            alpha = 0.15, inherit.aes = FALSE) +
  geom_line(aes(y = rt), linewidth = 0.8, color = "gray40") +
  geom_point(aes(y = rt, color = response_label), size = 2.5, alpha = 0.8) +
  scale_color_manual(values = unit_colors) +
  scale_fill_manual(values = c(`TRUE` = "white", `FALSE` = "lightcoral"), guide = "none") +
  labs(title = "Response Time",
       x = "Trial", y = "RT (s)", color = "Response") +
  theme_minimal()

# Plot activations
act_long <- df %>%
  pivot_longer(cols = starts_with("act_"),
               names_to = "unit",
               names_prefix = "act_",
               values_to = "activation") %>%
  mutate(unit = factor(unit, levels = unit_levels))  # enforce legend order


p_act <- ggplot(act_long, aes(x = trial, y = activation, color = unit)) +
  geom_rect(data = band_df,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = congruent),
            alpha = 0.15, inherit.aes = FALSE) +
  geom_line(linewidth = 0.8, alpha = 0.8) +
  scale_color_manual(values = unit_colors) +
  scale_fill_manual(values = c(`TRUE` = "white", `FALSE` = "lightcoral"), guide = "none") +
  labs(title = "Output Activations", x = "Trial", y = "Activation", color = "Unit") +
  theme_minimal()


# Plot control signals
ctrl_long <- df %>%
  pivot_longer(cols = starts_with("ctrl_"),
               names_to = "unit",
               names_prefix = "ctrl_",
               values_to = "control") %>%
  mutate(unit = factor(unit, levels = unit_levels))  # enforce legend order


p_ctrl <- ggplot(ctrl_long, aes(x = trial, y = control, color = unit)) +
  geom_rect(data = band_df,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = congruent),
            alpha = 0.15, inherit.aes = FALSE) +
  geom_line(linewidth = 0.8, alpha = 0.8) +
  scale_color_manual(values = unit_colors) +
  scale_fill_manual(values = c(`TRUE` = "white", `FALSE` = "lightcoral"), guide = "none") +
  labs(title = "Control Signals", x = "Trial", y = "Control Level", color = "Unit") +
  theme_minimal()

(p_acc + p_rt) / (p_act + p_ctrl)

final_plot <- (p_acc + p_rt) / (p_act + p_ctrl)
ggsave("plots/full_dynamics_plot.png", final_plot, width = 15, height = 7, dpi = 300)


# Add drift rates for plotting against activations
v_mat <- t(sapply(activation_list, function(a) v_base * a))  # n_trials x 4
colnames(v_mat) <- paste0("v_", c("dog", "bird", "cat", "fish"))
df <- cbind(dat, v_mat)
names(df)
str(df)
v_long <- df %>%
  pivot_longer(cols = starts_with("v_"),
               names_to = "unit",
               names_prefix = "v_",
               values_to = "drift") %>%
  mutate(unit = factor(unit, levels = unit_levels))  # enforce order

p_drift <- ggplot(v_long, aes(x = trial, y = drift, color = unit)) +
  geom_rect(data = band_df,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = congruent),
            alpha = 0.15, inherit.aes = FALSE) +
  geom_line(linewidth = 0.8, alpha = 0.8) +
  scale_color_manual(values = unit_colors) +
  scale_fill_manual(values = c(`TRUE` = "white", `FALSE` = "lightcoral"), guide = "none") +
  labs(title = "Drift Rates", x = "Trial", y = "Drift rate", color = "Unit") +
  theme_minimal()

p_act_clean <- p_act + theme(axis.title.x = element_blank(),
                             axis.text.x  = element_blank(),
                             axis.ticks.x = element_blank())
# Keep x-axis on bottom plot only
p_drift_clean <- p_drift

drift_activations_plot <- p_act_clean / p_drift_clean
ggsave("plots/drift_activations_plot.png", drift_activations_plot, width = 15, height = 5, dpi = 300)



# -------------------------------------------------------------------------

library(patchwork)

# Remove x-axis elements from upper plots
p_acc_clean <- p_acc + theme(axis.title.x = element_blank(),
                             axis.text.x  = element_blank(),
                             axis.ticks.x = element_blank())
p_rt_clean  <- p_rt  + theme(axis.title.x = element_blank(),
                             axis.text.x  = element_blank(),
                             axis.ticks.x = element_blank())
p_act_clean <- p_act + theme(axis.title.x = element_blank(),
                             axis.text.x  = element_blank(),
                             axis.ticks.x = element_blank())
# Keep x-axis on bottom plot only
p_ctrl_clean <- p_ctrl

# Combine with shared layout
final_plot_vertical <- (
  p_acc_clean /
    p_rt_clean /
    p_act_clean /
    p_ctrl_clean
) +
  plot_layout(guides = "collect", heights = c(1, 1, 1, 1)) & 
  theme(plot.margin = margin(5, 10, 5, 10))  # optional consistent margin

# Save
ggsave("plots/full_dynamics_plot_vertical.png",
       final_plot_vertical,
       width = 16, height = 9, dpi = 300)



# Fit model ---------------------------------------------------------------

start_par <- c(v_base = 1, A = 0.3, B = 0.3, t0 = 0.25, sv = 0.2)

fit <- optim(
  par = start_par,
  fn = neg_log_lik,
  method = "Nelder-Mead",  # Or try "L-BFGS-B" with bounds
  trial_data = dat,
  activation_list = activation_list,
  control = list(maxit = 5000)
)

fit$par  # Estimated parameters
fit$value  # Final negative log-likelihood

true_pars <- c(v_base = 1, A = 0.5, B = 0.4, t0 = 0.3, sv = 0.2)
recovered_pars <- fit$par
round(rbind(True = true_pars, Recovered = recovered_pars), 3)


library(parallel)

fit_lba_multistart_parallel <- function(n_starts, par_bounds, trial_data, activation_list, seed = 1) {
  set.seed(seed)
  
  start_list <- replicate(n_starts, runif(5, min = par_bounds$lower, max = par_bounds$upper), simplify = FALSE)
  
  fits <- mclapply(seq_len(n_starts), function(i) {
    start_par <- start_list[[i]]
    fit <- tryCatch(
      optim(
        par = start_par,
        fn = neg_log_lik,
        method = "L-BFGS-B",
        lower = par_bounds$lower,
        upper = par_bounds$upper,
        trial_data = trial_data,
        activation_list = activation_list,
        control = list(maxit = 5000)
      ),
      error = function(e) NULL
    )
    if (!is.null(fit)) fit$start <- start_par
    return(fit)
  }, mc.cores = detectCores() - 1)
  
  # Filter successful fits
  valid_fits <- Filter(function(x) !is.null(x) && is.finite(x$value), fits)
  return(valid_fits)
}

par_bounds <- list(
  lower = c(v_base = 0.01, A = 0.1, B = 0.1, t0 = 0.1, sv = 0.01),
  upper = c(v_base = 5.0, A = 2.0, B = 2.0, t0 = 1.0, sv = 1.0)
)

fits <- fit_lba_multistart_parallel(
  n_starts = 10,
  par_bounds = par_bounds,
  trial_data = dat,
  activation_list = activation_list
)

likelihoods <- sapply(fits, function(fit) fit$value)
best_idx <- which.min(likelihoods)

# Extract parameter estimates from all valid fits
fit_params <- do.call(rbind, lapply(fits, function(fit) {
  out <- fit$par
  names(out) <- c("v_base", "A", "B", "t0", "sv")
  out
}))
fit_df <- as.data.frame(fit_params)
fit_df$fit_id <- seq_len(nrow(fit_df))

fit_long <- pivot_longer(
  fit_df,
  cols = -fit_id,
  names_to = "parameter",
  values_to = "estimate"
)

true_pars <- c(v_base = 1, A = 0.5, B = 0.4, t0 = 0.3, sv = 0.2)
true_df <- data.frame(
  parameter = names(true_pars),
  true_value = as.numeric(true_pars)
)

ggplot(fit_long, aes(x = parameter, y = estimate)) +
  geom_jitter(width = 0.1, height = 0, alpha = 0.6, color = "steelblue") +
  geom_point(data = true_df, aes(x = parameter, y = true_value),
             color = "red", shape = 18, size = 3) +
  labs(title = "Parameter Recovery Across Multi-Start Fits",
       y = "Estimated Value", x = "Parameter") +
  theme_minimal()

# Merge fit estimates with true values
fit_errors <- fit_long %>%
  left_join(true_df, by = "parameter") %>%
  mutate(error = estimate - true_value)

fit_errors %>%
  group_by(parameter) %>%
  summarise(
    mean_error = mean(error),
    sd_error   = sd(error),
    min_error  = min(error),
    max_error  = max(error)
  )

ggplot(fit_errors, aes(x = parameter, y = error)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  geom_jitter(width = 0.1, height = 0, alpha = 0.6, color = "darkorange") +
  geom_boxplot(width = 0.3, outlier.shape = NA, fill = "gray90", alpha = 0.5) +
  labs(title = "Parameter Estimation Errors Across Fits",
       y = "Error (Estimate - True Value)", x = "Parameter") +
  theme_minimal()
