data {
  int N;
  int<lower=0,upper=1> x[N];
}
parameters {
  real<lower=0,upper=1> z;
}
model {
  z ~ beta(1, 1);
  x ~ bernoulli(z);
}

guide parameters {
 real<lower=0> alpha_q;
 real<lower=0> beta_q;
}

guide {
  z ~ beta(alpha_q, beta_q);
}