data {
  int<lower=0,upper=1> x[10];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
    real psi;
  theta ~ uniform(0,1);
  psi ~ uniform(0,1);       // <- sampling a non random variable

  for (i in 1:10)
    x[i] ~ bernoulli(theta);
}
