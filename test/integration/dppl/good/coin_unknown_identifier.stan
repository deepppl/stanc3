data {
  int<lower=0,upper=1> x[10];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ uniform(0,1);
  th2 ~ uniform(0,1);  // <- unknown identifier

  for (i in 1:10)
    x[i] ~ bernoulli(theta);
}
