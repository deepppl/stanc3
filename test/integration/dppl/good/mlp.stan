networks { vector mlp(int[,,] imgs); }
data {
 int batch_size; int nx; int nh; int ny;
 int <lower=0, upper=1> imgs[28,28,batch_size];
 int <lower=1, upper=10> labels[batch_size];
}
parameters {
  real mlp.l1.weight[nh, nx];
  real mlp.l1.bias[nh];
  real mlp.l2.weight[ny, nh];
  real mlp.l2.bias[ny];
}
model {
  vector[batch_size] lambda;
  mlp.l1.weight ~  normal(0, 1);
  mlp.l1.bias ~ normal(0, 1);
  mlp.l2.weight ~ normal(0, 1);
  mlp.l2.bias ~  normal(0, 1);
  lambda = mlp(imgs);
  labels ~ categorical_logit(lambda);
}
guide parameters {
  real w1_mu[nh, nx];
  real w1_sgma[nh, nx];
  real b1_mu[nh];
  real b1_sgma[nh];
  real w2_mu[ny, nh];
  real w2_sgma[ny, nh];
  real b2_mu[ny];
  real b2_sgma[ny];
}
guide {
  mlp.l1.weight ~ normal(w1_mu, exp(w1_sgma));
  mlp.l1.bias   ~ normal(b1_mu, exp(b1_sgma));
  mlp.l2.weight ~ normal(w2_mu, exp(w2_sgma));
  mlp.l2.bias   ~ normal(b2_mu, exp(b2_sgma));
}
