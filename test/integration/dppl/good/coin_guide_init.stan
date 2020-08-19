data {
  int<lower=0,upper=1> x[10];
}
parameters {
  real<lower=0.0,upper=1.0> theta;
}


model {
  theta ~ beta(10.0, 10.0);
  for (i in 1:10)
    x[i] ~ bernoulli(theta);
}


guide parameters
{
  real<lower=0>  alpha_q = tensor(15.0);
  real<lower=0>  beta_q = tensor(15.0);
}

guide {
  theta ~ beta(alpha_q, beta_q);
}
