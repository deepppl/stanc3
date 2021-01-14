data {
}
parameters {
  real<lower=0> theta;
}
model {
  log(theta) ~ normal(log(10.0), 1.0);
  target += -log(fabs(theta));
}
