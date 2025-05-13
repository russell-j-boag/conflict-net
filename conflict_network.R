rm(list=ls())

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

# Default weights as vectors

WHt <- c(3, -3, -3, -3, -3, 3, -3, -3, -3, -3, 3, -3, -3, -3, -3, 3)
WHd <- c(3.5, -3.5, -3.5, -3.5, -3.5, 3.5, -3.5, -3.5, -3.5, -3.5, 3.5, -3.5, -3.5, -3.5, -3.5, 3.5)

WOt <- c(1.5, -1.5, -1.5, -1.5, -1.5, 1.5, -1.5, -1.5, -1.5, -1.5, 1.5, -1.5, -1.5, -1.5, -1.5, 1.5)
WOd <- c(1.5, -1.5, -1.5, -1.5, -1.5, 1.5, -1.5, -1.5, -1.5, -1.5, 1.5, -1.5, -1.5, -1.5, -1.5, 1.5)

# Default weights as matrices
wHt <- matrix(WHt,nrow=4,byrow=T)
wHd <- matrix(WHd,nrow=4,byrow=T)
wOt <- matrix(WOt,nrow=4,byrow=T)
wOd <- matrix(WOd,nrow=4,byrow=T)

# Default bias vectors
bHt <- c(-4, -4, -4, -4)
bHd <- c(-4, -4, -4, -4) 

# Task weights (only distractor gets used.)
wTt <- c(2, 2, 2, 2)
wTd <- c(2, 2, 2, 2)

# Items (lower case is target, upper case is distractor)
dog <- DOG <- c(1,0,0,0)
bird <- BIRD <- c(0,1,0,0)
cat <- CAT <- c(0,0,1,0)
fish <- FISH <- c(0,0,0,1)

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


#### Functions ----

# Net input function, W = weight matrix, ia = input activation
net <- function(W,ia) W %*% ia

# Calculate conflict
get_conflict <- function(aO,nd) -sum(aO*nd)  

# Control function
control_update <- function(control_val, conflict,
                           lambda = 0.5, alpha = 0.8) {
  (1 - lambda) * control_val + lambda * alpha * conflict
}

decay_non_target_activation <- function(aO, target_id, decay_rate) {
  aO[-target_id] <- aO[-target_id] * (1 - decay_rate)
  aO
}

# control_update_original <- function(target,control,conflict,
#                            lambda=.5,alpha=.8) {
#   (1-lambda)*control[as.logical(target)] +
#   lambda*alpha*conflict[as.logical(target)]
# }

# between_update_original <- function(stimulus,state,nn,
#                            lambda=.5,alpha=.8,
#                            print_conflict=FALSE) 
# # full update of nn state based on target and distractor 
# {
#   
#     # Calculate new hidden net
#     state$nHt <- net(nn$wHt,stimulus$target)     + nn$bHt
#     state$nHd <- net(nn$wHd,stimulus$distractor) + nn$bHd
#     # Update hidden activation
#     state$aHt <- nn$activation_fun(state$nHt)
#     state$aHd <- nn$activation_fun(state$nHd)
#  
#   # Update output
#     # Calculate net to output units
#     state$nOt <- net(nn$wOt,state$aHt)
#     
#     # CLARIFY IF BELOW SHOULD APPLY THE TARGET OR DISTRACTOR WEIGHTS?
#     state$nOd <- net(nn$wOt, # includes control signal for target item
#       state$aHd+state$control[as.logical(stimulus$target)]*nn$wTd)
#     state$aO <- nn$activation_fun(state$nOt+state$nOd)
# 
#   # Calculate conflict
#   conflict <- get_conflict(state$aO,state$aHd)
# 
#   if(print_conflict) print(conflict)
#   
#   # Update item-specific control
#   state$control[as.logical(stimulus$target)] <- 
#     control_update(stimulus$target,state$control,conflict,
#                    lambda=lambda,alpha=alpha)
#   
#   return(state)
#   
# }

# Between-trial update
between_update <- function(stimulus, state, nn,
                           lambda = 0.5, alpha = 0.8, 
                           activation_decay = 0.15,
                           control_decay = 0.15,
                           stim_lookup,
                           print_conflict = FALSE) 
{
  # Look up one-hot encoded target and distractor vectors
  target_vec     <- stim_lookup[[stimulus$target]]
  distractor_vec <- stim_lookup[[stimulus$distractor]]
  
  # Compute hidden net inputs
  state$nHt <- net(nn$wHt, target_vec) + nn$bHt
  state$nHd <- net(nn$wHd, distractor_vec) + nn$bHd
  
  # Hidden layer activations
  state$aHt <- nn$activation_fun(state$nHt)
  state$aHd <- nn$activation_fun(state$nHd)
  
  # Output net input from target
  state$nOt <- net(nn$wOt, state$aHt)
  
  # Identify which item is active (e.g., "dog", "cat")
  target_name <- stimulus$target
  distractor_name <- stimulus$distractor
  
  # # Print debug info
  # cat("Target:", target_name, "\n")
  # print(target_vec)
  # cat("Distractor:", distractor_name, "\n")
  # print(distractor_vec)
  # cat("Control state:\n")
  # print(state$control)
  
  # Output net input from distractor + control modulation
  ctrl_vec  <- state$control[target_name] * as.numeric(nn$wTd)
  ctrl_term <- matrix(ctrl_vec, nrow = 4, ncol = 1)
  input_to_od <- matrix(as.numeric(state$aHd), nrow = 4, ncol = 1) + ctrl_term
  state$nOd <- net(nn$wOd, input_to_od)
  
  # Final output activation
  state$aO <- nn$activation_fun(state$nOt + state$nOd)
  
  # Decay output activation for non-target units toward zero
  target_id <- which.max(stim_lookup[[target_name]])
  state$aO <- decay_non_target_activation(state$aO, target_id, activation_decay)
 
  # Compute conflict
  conflict <- get_conflict(state$aO, state$aHd)
  if (print_conflict) print(conflict)
  
  # Passive decay for non-target items
  non_target_names <- setdiff(names(state$control), target_name)
  state$control[non_target_names] <- state$control[non_target_names] * (1 - control_decay)
  
  # Conflict-based update for the target (same as before)
  state$control[target_name] <- control_update(
    control_val = state$control[target_name],
    conflict = conflict,
    lambda = lambda,
    alpha = alpha
  )
  
  return(state)
}

# Within-trial update
within_update <- function(stimulus,state,nn,
                         tau=.1,lambda=.5,alpha=.8,
                         print_conflict=FALSE) 
# gradual update of nn state based on target and distractor 
{
  
    # Store old net
    nHt_1 <- state$nHt
    nHd_1 <- state$nHd
    # Calculate new hidden net
    state$nHt <- net(nn$wHt,stimulus$target)     + nn$bHt
    state$nHd <- net(nn$wHd,stimulus$distractor) + nn$bHd
    
    # Update hidden activation
    tauc <- 1-tau
    state$aHt <- nn$activation_fun(tauc*nHt_1 + tau*state$nHt)
    state$aHd <- nn$activation_fun(tauc*nHd_1 + tau*state$nHd)
    
    # Store old net
    nOt_1 <- state$nOt
    nOd_1 <- state$nOd
    # Calculate net to output units
    state$nOt <- net(nn$wOt,state$aHt)
    state$nOd <- net(nn$wOt, # includes control signal for target item
      state$aHd+state$control[as.logical(stimulus$target)]*nn$wTd)
    state$aO <- nn$activation_fun(tauc*(nOt_1+nOd_1) + 
                            tau*(state$nOt+state$nOd))
  # Calculate conflict
  conflict <- get_conflict(state$aO,state$aHd)

  if(print_conflict) print(conflict)
  
  # Update item-specific control
  state$control[as.logical(stimulus$target)] <- 
    control_update(stimulus$target,state$control,conflict,
                   lambda=lambda,alpha=alpha)
  
  return(state)
  
}

get_stimulus <- function(name) {
  if (!exists(name, mode = "numeric")) 
    stop(paste("Stimulus", name, "not found"))
  get(name)
}

# Run between-trial updates
run_between_sequence <- function(stim_sequence, state0, nn, stim_lookup, ...) {
  n_trials <- nrow(stim_sequence)
  states <- vector("list", n_trials + 1)
  states[[1]] <- state0
  
  for (i in 1:n_trials) {
    stim <- list(
      target = stim_sequence$target[i],
      distractor = stim_sequence$distractor[i]
    )
    states[[i + 1]] <- between_update(
      stimulus = stim,
      state = states[[i]],
      nn = nn,
      stim_lookup = stim_lookup,
      ...
    )
  }
  
  return(states)
}

# run_between_original <- function(ss, state0, nn, stim_lookup) {
#   slist <- apply(ss, 1, \(x) list(
#     target = stim_lookup[[x[1]]],
#     distractor = stim_lookup[[x[2]]]
#   ))
#   states <- vector(mode = "list", length = length(slist) + 1)
#   states[[1]] <- state0
#   for (i in 2:length(states)) {
#     states[[i]] <- between_update(slist[[i - 1]], state = states[[i - 1]], nn = nn)
#   }
#   states
# }

# Run within-trial updates
run_within <- function(n,stimulus,state0,nn,tau=.01) 
{
  states <- vector(mode="list",length=n+1)
  states[[1]] <- state0
  for (i in 2:(n+1)) {
    states[[i]] <- within_update(stimulus,states[[i-1]],nn,tau=tau)
  }
  states
}

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

# Test within updates
# cwithin <- run_within(100,cong,state0,nn,tau=.001)
# iwithin <- run_within(100,incong,state0,nn)
# 
# states <- cwithin
# pick <- "aO"
# do.call(rbind,lapply(states,\(x) t(x[[pick]])))


#### Simulate a trial sequence ----

set.seed(8)
n_per_target <- 40

# Make targets
targets <- rep(c("dog", "bird", "cat", "fish"), each = n_per_target)

# Make distractors
distractors <- character(length(targets))
for (i in seq_along(targets)) {
  if (runif(1) < 0.75) {
    # 75% of the time, distractor matches target
    distractors[i] <- toupper(targets[i])
  } else {
    # 25% of the time, distractor is randomly drawn from other items
    distractors[i] <- sample(setdiff(toupper(c("dog", "bird", "cat", "fish")), toupper(targets[i])), 1)
  }
}

stim_sequence <- data.frame(
  target = targets,
  distractor = distractors,
  stringsAsFactors = FALSE
)
str(stim_sequence)
states <- run_between_sequence(stim_sequence, state0, nn, stim_lookup)
states


# Skip the initial state (index 1), extract aO from each trial
activation_list <- lapply(states[-1], function(s) s$aO)
control_list <- lapply(states[-1], function(s) s$control)
save(activation_list, control_list, file = "data/activation_list.RData")


# LBA parameters (true values for simulation)
v_base <- 1
A <- 0.5
B <- 0.6
b <- A + B
t0 <- 0.3
sv <- 0.2

# Scale activations to drift rates
v_mat <- t(sapply(activation_list, function(a) v_base * a))  # n_trials x n_acc
n_acc = ncol(v_mat)
n_trials = nrow(v_mat)

# Simulate each trial separately
sim_data <- do.call(rbind, lapply(1:n_trials, function(i) {
  rlba_norm(
    n = 1,
    A = A,
    b = rep(b, n_acc),
    t0 = t0,
    mean_v = v_mat[i, ],
    sd_v = rep(sv, n_acc),
    posdrift = TRUE
  )
}))

sim_data <- data.frame(sim_data)

sim_full <- cbind(
  trial = as.numeric(seq_len(length(activation_list))),
  stim_sequence,
  response = sim_data$response,
  rt = sim_data$rt
)

sim_full$congruent <- tolower(sim_full$target) == tolower(sim_full$distractor)

# Map target names to response numbers
stim_map <- c("dog" = 1, "bird" = 2, "cat" = 3, "fish" = 4)
sim_full$correct <- stim_map[sim_full$target] == sim_full$response
sim_full <- sim_full[,c("target", "distractor", "congruent", "response", "rt", "correct")]
head(sim_full)
tail(sim_full)
summary(sim_full$correct)
table(sim_full$congruent)
mean(sim_full$correct)
dat <- sim_full

# Save
save(dat, file = "data/sim_dat.RData")


# Add trial activations and conflict signals
str(activation_list)
str(control_list)

# Convert each [4×1] matrix to a vector
activation_matrix <- do.call(rbind, lapply(activation_list, function(x) as.numeric(x)))
control_matrix <- do.call(rbind, lapply(control_list, function(x) as.numeric(x)))

# Assign column names (match response units)
colnames(activation_matrix) <- c("act_dog", "act_bird", "act_cat", "act_fish")
colnames(control_matrix) <- c("ctrl_dog", "ctrl_bird", "ctrl_cat", "ctrl_fish")

# Combine with trial metadata
activation_df <- data.frame(
  trial = as.numeric(seq_len(length(activation_list))),
  stim_sequence,
  congruent = sim_full$congruent,
  response = sim_full$response,
  rt = sim_full$rt,
  correct = sim_full$correct,
  activation_matrix,
  control_matrix
)

head(activation_df)
tail(activation_df)
str(activation_df)

# Save
save(activation_df, file = "data/activation_df.RData")


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
activation_df <- activation_df %>%
  mutate(
    acc_numeric = as.numeric(correct),
    acc_ma = rollmean(acc_numeric, k = 5, fill = NA, align = "center")
  )

response_labels <- c("dog", "bird", "cat", "fish")
activation_df$response_label <- factor(response_labels[activation_df$response],
                                       levels = names(unit_colors))

# Rectangles for background bands
band_df <- activation_df %>%
  transmute(
    xmin = trial - 0.5,
    xmax = trial + 0.5,
    ymin = -Inf,
    ymax = Inf,
    congruent = congruent
  )

mean_acc <- mean(activation_df$acc_numeric, na.rm = TRUE)

# Plot accuracy
p_acc <- ggplot(activation_df, aes(x = trial)) +
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
p_rt <- ggplot(activation_df, aes(x = trial)) +
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
act_long <- activation_df %>%
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
ctrl_long <- activation_df %>%
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
