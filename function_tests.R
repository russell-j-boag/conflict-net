# Function tests
library("testthat")

# Calculate conflict
get_conflict <- function(aO,nd) -sum(aO*nd)  

test_that("get_conflict returns correct negative dot product", {
  aO <- c(0.1, 0.5, 0.2, 0.2)
  nd <- c(1, 0, 0, 0)
  expect_equal(get_conflict(aO, nd), -0.1)
  
  nd <- c(0, 1, 0, 0)
  expect_equal(get_conflict(aO, nd), -0.5)
  
  nd <- c(0.25, 0.25, 0.25, 0.25)
  expect_equal(get_conflict(aO, nd), -sum(aO * nd))
})

# Control function
control_update <- function(control_val, conflict,
                           lambda = 0.5, alpha = 0.8) {
  (1 - lambda) * control_val + lambda * alpha * conflict
}

test_that("control_update computes weighted sum correctly", {
  control_val <- 0.5
  conflict <- -0.2
  lambda <- 0.5
  alpha <- 0.8
  
  expected <- (1 - lambda) * control_val + lambda * alpha * conflict
  expect_equal(control_update(control_val, conflict, lambda, alpha), expected)
  
  # Edge cases
  expect_equal(control_update(0, 0, lambda, alpha), 0)
  expect_equal(control_update(1, 0, lambda, alpha), 0.5)
  expect_equal(control_update(0, 1, lambda, alpha), 0.4)
})
