data {
  int<lower=0,upper=1> x[10];
}

parameters {
  real<lower=0,upper=1> theta;
}

model {
  // theta ~ beta(10.0, 10.0);
  // for (i in 1:10)
  //   x[i] ~ bernoulli(theta);
}

guide parameters {
  real<lower=0>  alpha_q;
  real<lower=0>  beta_q;
}

guide {
  alpha_q = 15.0;
  beta_q = 15.0;
  theta ~ beta(alpha_q, beta_q);
}
