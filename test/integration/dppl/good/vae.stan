networks {
  real[,] decoder(real[] x);
  real[,] encoder(int[,] x);
}
data {
    int nz;
    int<lower=0, upper=1> x[28, 28];
}
parameters {
    real z[nz];
}
model {
  real mu[28, 28];
  z ~ normal(0, 1);
  mu = decoder(z);
  for (i in 1:28)
    x[i] ~ bernoulli(mu[i]);
}
guide {
  real encoded[2, nz] = encoder(x);
  real mu_z[nz] = encoded[1];
  real sigma_z[nz] = encoded[2];
  z ~ normal(mu_z, sigma_z);
}