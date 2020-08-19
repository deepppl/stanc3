data {
  int<lower=0,upper=1> x[10];
}
parameters {
  real<lower=0,upper=1> theta;
  real theta;   // <- Already declared variable>
}
model {
  theta ~ uniformWeird(0,1);

  for (i in 1:10)
    x[i] ~ bernoulli(theta);
}
