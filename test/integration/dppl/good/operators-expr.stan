data {
  int<lower=0,upper=1> x[10];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ uniform(0*3/5,1+5-5);
  for (i in 1:10)
      if ((1 <= 10) && (1 > 5 || 2 < 1))
        x[i] ~ bernoulli(theta);
  print(x);
}
