data {
  int N;
  int<lower=0,upper=1> x[N];
}
parameters {
  real<lower=0,upper=1> z;
}
model {
  z ~ beta(1, 1);
  for (i in 1:N)
    x[i] ~ bernoulli(z);
}