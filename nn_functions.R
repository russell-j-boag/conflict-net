#### Functions for Cohen, Dunbar, & McLelland network ----

# Net input function, W = weight matrix, ia = input activation
net <- function(W,ia) W %*% ia

# Calculate conflict
# get_conflict <- function(aO,nd) -sum(aO*nd)  
get_conflict <- function(aO, nd) {
  dot_product <- sum(aO * nd)
  norm_aO <- sqrt(sum(aO^2))
  norm_nd <- sqrt(sum(nd^2))
  # Cosine similarity
  cosine_similarity <- dot_product / (norm_aO * norm_nd)
  # Negative of cosine similarity
  conflict <- cosine_similarity - 1
  return(conflict)
}


# Control function
control_update <- function(control_val, conflict,
                           lambda = 0.5, alpha = 0.8) {
  (1 - lambda) * control_val + lambda * alpha * conflict
}

# Non-target activation decay
decay_non_target_activation <- function(aO, target_id, decay_rate) {
  aO[-target_id] <- aO[-target_id] * (1 - decay_rate)
  aO
}

# Between-trial update
between_update <- function(stimulus, state, nn,
                           lambda = 0.5, alpha = 0.8, 
                           activation_decay = 0,
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
  state$conflict <- conflict
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

# Make weight matrix with positive diagonal and negative off-diagonal
make_weight_matrix <- function(n_units = 4, pos = 1, neg = -1) {
  mat <- matrix(neg, nrow = n_units, ncol = n_units)
  diag(mat) <- pos
  return(mat)
}
?fit

simulate_lba_from_states <- function(states, stim_sequence,
                                    v_int = 0.2, v_scale = 1, A = 0.5, B = 0.5, t0 = 0.3, sv = 0.2) {
  # Check dimensions
  if (length(states) != nrow(stim_sequence) + 1) {
    stop("Number of states should be one more than the number of trials.")
  }
  
  # Drop initial state and extract activations and controls
  activation_list <- lapply(states[-1], function(s) s$aO)
  control_list    <- lapply(states[-1], function(s) s$control)
  conflict_list <- lapply(states[-1], function(s) s$conflict)
  
  # Convert to drift matrix
  v_mat <- t(sapply(activation_list, function(a) v_int + v_scale * a))  # n_trials x n_acc
  n_acc <- ncol(v_mat)
  n_trials <- nrow(v_mat)
  b <- A + B
  
  # Simulate LBA responses
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
  
  # Add derived trial info
  sim_full <- cbind(
    trial = seq_len(n_trials),
    stim_sequence,
    response = sim_data$response,
    rt = sim_data$rt
  )
  
  # Add congruency and accuracy
  sim_full$congruent <- tolower(sim_full$target) == tolower(sim_full$distractor)
  stim_map <- c("dog" = 1, "bird" = 2, "cat" = 3, "fish" = 4)
  sim_full$correct <- stim_map[sim_full$target] == sim_full$response
  
  # Convert activation/control lists to matrices
  activation_matrix <- do.call(rbind, lapply(activation_list, as.numeric))
  control_matrix    <- do.call(rbind, lapply(control_list, as.numeric))
  conflict_matrix    <- do.call(rbind, lapply(conflict_list, as.numeric))
  
  # Name columns
  colnames(activation_matrix) <- paste0("act_", names(stim_map))
  colnames(control_matrix)    <- paste0("ctrl_", names(stim_map))
  colnames(conflict_matrix) <- "conflict"
  
  # Add to sim_full
  sim_full <- cbind(sim_full, conflict_matrix, control_matrix, activation_matrix)
  
  return(sim_full)
}


# LBA likelihood function
library(rtdists)
neg_log_lik <- function(par, trial_data, activation_list) {
  v_int   <- par[1]
  v_scale <- par[2]
  A       <- par[3]
  B       <- par[4]
  t0      <- par[5]
  sv_val  <- par[6]
  
  if (any(!is.finite(par)) || any(par <= 0)) return(1e6)
  
  b_val <- A + B
  if (b_val <= A) return(1e6)
  
  total_nll <- 0
  for (i in seq_along(trial_data$rt)) {
    rt   <- trial_data$rt[i]
    resp <- trial_data$response[i]
    act  <- activation_list[[i]]
    
    mean_v <- v_int + v_scale * act
    b <- rep(b_val, 4)
    sv <- rep(sv_val, 4)
    
    dens <- tryCatch(
      dlba_norm(rt, A, b[resp], t0, mean_v[resp], sv[resp], posdrift = TRUE),
      error = function(e) NA
    )
    
    survs <- tryCatch(
      sapply(setdiff(1:4, resp), function(j) {
        1 - plba_norm(rt, A, b[j], t0, mean_v[j], sv[j], posdrift = TRUE)
      }),
      error = function(e) rep(NA, 3)
    )
    
    if (is.na(dens) || any(is.na(survs)) || dens <= 0 || any(survs <= 0)) {
      return(1e6)
    }
    
    log_lik <- log(dens) + sum(log(survs))
    total_nll <- total_nll - log_lik
  }
  
  return(total_nll)
}

