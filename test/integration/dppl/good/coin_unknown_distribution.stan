data {
  int<lower=0,upper=1> x[10];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ uniformWeird(0,1);   //<- unknown distribution

  for (i in 1:10)
    x[i] ~ bernoulli(theta);
}
