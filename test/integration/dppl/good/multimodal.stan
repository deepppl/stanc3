parameters {
  real cluster;
  real theta;
}
model {
  real mu;
  cluster ~ normal(0, 1);
  if (cluster > 0) { mu = 2; }
  else { mu = 0; }
  theta ~ normal(mu, 1);
}