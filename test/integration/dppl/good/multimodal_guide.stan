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

guide parameters {
  real mu_cluster;
  real mu1; real mu2;
  real log_sigma1; real log_sigma2;
}

guide {
  cluster ~ normal(mu_cluster, 1);
  if (cluster > 0) {
    theta ~ normal(mu1, exp(log_sigma1));
  } else {
    theta ~ normal(mu2, exp(log_sigma2));
  }
}