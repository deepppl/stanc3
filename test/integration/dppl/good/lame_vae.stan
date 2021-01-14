networks {
  Decoder decoder;
  Encoder encoder;
}
data {
  int x;
  int nz;
  int batch_size;
}
parameters {
  int latent;
}
model {
  int loc_img;
  latent ~ normal(zeros(nz), ones(nz), batch_size);
  loc_img = decoder(latent);
  x ~ bernoulli(loc_img);
}
guide {
  real encoded[2];
  real mu;
  real sigma;
  encoded = encoder(x);
  mu = encoded[1];
  sigma = encoded[2];
  latent ~ Normal(mu, sigma, batch_size);
}