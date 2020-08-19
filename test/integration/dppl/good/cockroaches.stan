data {
  int<lower=0> N; 
  vector[N] exposure2;
  vector[N] roach1;
  vector[N] senior;
  vector[N] treatment;
  int y[N];
}
transformed data {
  vector[N] log_expo;
  vector[N] sqrt_roach;

  log_expo = log(exposure2);
  sqrt_roach = sqrt(roach1);
}
parameters {
  real beta_1;
  real beta_2;
  real beta_3;
  real beta_4;
} 
model {
  beta_1 ~ normal(0, 5);
  beta_2 ~ normal(0, 2.5);
  beta_3 ~ normal(0, 2.5);
  beta_4 ~ normal(0, 2.5);
  y ~ poisson_log(log_expo + beta_1 + beta_2 * sqrt_roach + beta_3 * treatment
                  + beta_4 * senior);
}