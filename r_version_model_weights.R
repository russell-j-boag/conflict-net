# model weights
init_model <- function(n_resp, hw=3, dhwb=0.5, ow=1.5, dowb=0, task_weights=2, bias=-4) {
    # Creates base matrix of activation weights on the diagonal and inhibitory weights on the off-diagonals
    base_weights <- 2 * diag(n_resp) - matrix(1, n_resp, n_resp)
    
    # Create model list (equivalent to Python dictionary)
    model <- list(
        targetHiddenWeights = base_weights * hw,
        distractorHiddenWeights = base_weights * (hw + dhwb),
        targetOutputWeights = base_weights * ow,
        distractorOutputWeights = base_weights * (ow + dowb),
        targetTaskWeights = matrix(task_weights, nrow=n_resp, ncol=1),
        distractorTaskWeights = matrix(task_weights, nrow=n_resp, ncol=1),
        targetHiddenBias = matrix(bias, nrow=n_resp, ncol=1),
        distractorHiddenBias = matrix(bias, nrow=n_resp, ncol=1)
    )
    
    return(model)
}

model_output <- init_model(n_resp=4, hw=3, dhwb=0.5, ow=1.5, dowb=0, task_weights=2, bias=-4)

model_df <- data.frame(
    Component = names(model_output),
    Values = sapply(model_output, toString)
)

write.csv(model_df, "model_weights.csv", row.names=FALSE)

model_df$Values

















