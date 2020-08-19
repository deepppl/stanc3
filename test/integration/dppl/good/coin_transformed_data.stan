data {
  int<lower=0,upper=1> x[10];
}
transformed data {
  int<lower=0, upper=1> y[10];
  for (i in 1:10)
    y[i] = 1 - x[i];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ uniform(0,1);
  for (i in 1:10)
    y[i] ~ bernoulli(theta);
}
