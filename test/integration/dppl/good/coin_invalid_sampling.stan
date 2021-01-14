data {
  int<lower=0,upper=1> x[10];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta*2 ~ uniform(0,1);     // <- arbitrary expression as lhs of sampling not supported

  for (i in 1:10)
    x[i] ~ bernoulli(theta);
}
