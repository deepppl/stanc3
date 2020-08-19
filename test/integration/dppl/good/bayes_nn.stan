// Figure 6
networks {
  MLP mlp;
}
data {
  int<lower=0, upper=1> img[28, 28];
  int<lower=0, upper=9> label;
}
parameters {
  real mlp.l1.weight[_];
  real mlp.l1.bias[_];
  real mlp.l2.weight[_];
  real mlp.l2.bias[_];
}
model {
  real logits[10];
  mlp.l1.weight ~ normal(0, 1);
  mlp.l1.bias   ~ normal(0, 1);
  mlp.l2.weight ~ normal(0, 1);
  mlp.l2.bias   ~ normal(0, 1);
  logits = mlp(img);
  label ~ categorical_logits(logits);
}
guide parameters {
  real w1_mu[_]; real w1_sgma[_];
  real b1_mu[_]; real b1_sgma[_];
  real w2_mu[_]; real w2_sgma[_];
  real b2_mu[_]; real b2_sgma[_];
}
guide {
  mlp.l1.weight ~ normal(w1_mu, exp(w1_sgma));
  mlp.l1.bias   ~ normal(b1_mu, exp(b1_sgma));
  mlp.l2.weight ~ normal(w2_mu, exp(w2_sgma));
  mlp.l2.bias   ~ normal(b2_mu, exp(b2_sgma));
}