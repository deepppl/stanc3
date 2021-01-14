data {
}
parameters {
  real theta;
}
model {
  theta ~ normal(1000.0, 1.0);
  theta ~ normal(1000.0, 1.0);
}
