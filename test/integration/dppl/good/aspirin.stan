data {
  int N;
  vector[N] y;
  vector[N] s;
  real mu_loc;
  real mu_scale;
  real tau_scale;
  real tau_df;
}
parameters {
  vector[N] theta_raw;
  real mu;
  real tau;
}
transformed parameters {
  vector[N] theta;
  theta = tau * theta_raw + mu;
}
model {
  mu ~ normal(mu_loc, mu_scale);
  tau ~ student_t(tau_df, 0., tau_scale);
  theta_raw ~ normal(0., 1.);
  y ~ normal(theta, s);
}
generated quantities {
  vector[N] shrinkage;
  real tau2;
  tau2 = pow(tau, 2.);
  for (i in 1:N) {
    real v;
    v = pow(s[i], 2);
    shrinkage[i] = v / (v + tau2);
  }
}
